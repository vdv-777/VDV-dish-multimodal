# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
import pandas as pd
from tqdm import tqdm
import numpy as np
from dataset import CalorieDataset, get_transforms
from model import MultimodalCalorieModel

def prepare_ingredient_mapping(ingredients_df, dish_df):
    ingr_id_to_name = dict(zip(ingredients_df["id"], ingredients_df["ingr"]))
    
    def id_list_to_names(ingr_str):
        if pd.isna(ingr_str):
            return []
        ids = ingr_str.split(";")
        names = []
        for iid in ids:
            if iid.startswith("ingr_"):
                try:
                    num = int(iid.replace("ingr_", ""))
                    name = ingr_id_to_name.get(num)
                    if name:
                        names.append(name)
                except:
                    continue
        return names
    
    dish_df = dish_df.copy()
    dish_df["ingredient_names"] = dish_df["ingredients"].apply(id_list_to_names)
    dish_df["ingredient_names_str"] = dish_df["ingredient_names"].apply(lambda x: ",".join(x))
    
    all_names = set()
    for names in dish_df["ingredient_names"]:
        all_names.update(names)
    ingr_to_idx = {name: idx + 1 for idx, name in enumerate(sorted(all_names))}
    ingr_to_idx["<PAD>"] = 0

    return dish_df, ingr_to_idx

def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    ingredients_df = pd.read_csv(config.INGREDIENTS_PATH)
    dish_df = pd.read_csv(config.DISH_PATH)

    dish_df, ingr_to_idx = prepare_ingredient_mapping(ingredients_df, dish_df)

    if config.DEBUG:
        dish_df = dish_df.sample(n=min(config.DEBUG_SIZE, len(dish_df)), random_state=config.SEED).reset_index(drop=True)

    train_df = dish_df[dish_df["split"] == "train"]
    val_df = dish_df[dish_df["split"] == "test"]

    transforms_train = get_transforms(config, is_train=True)
    transforms_val = get_transforms(config, is_train=False)

    train_dataset = CalorieDataset(train_df, ingr_to_idx, config.IMAGES_DIR, transforms_train, config.MAX_INGREDIENTS, mode="train")
    val_dataset = CalorieDataset(val_df, ingr_to_idx, config.IMAGES_DIR, transforms_val, config.MAX_INGREDIENTS, mode="train")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = MultimodalCalorieModel(config, num_ingredients=len(ingr_to_idx)).to(device)
    model.freeze_image_encoder(unfreeze_patterns=config.IMAGE_UNFREEZE_LAYERS or [])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.HuberLoss(delta=100.0)

    mae_metric = MeanAbsoluteError().to(device)
    mse_metric = MeanSquaredError().to(device)

    best_val_mae = float('inf')

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for image, ingr, calories, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            image, ingr, calories = image.to(device), ingr.to(device), calories.to(device)
            optimizer.zero_grad()
            pred, _, _ = model(image, ingr)
            loss = criterion(pred, calories)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            mae_metric(pred, calories)

        train_mae = mae_metric.compute()
        mae_metric.reset()

        model.eval()
        val_dish_ids, val_preds, val_trues = [], [], []
        with torch.no_grad():
            for image, ingr, calories, mass, dish_id in val_loader:
                image, ingr, calories = image.to(device), ingr.to(device), calories.to(device)
                pred, _, _ = model(image, ingr)
                mae_metric(pred, calories)
                mse_metric(pred, calories)
                val_preds.extend(pred.cpu().numpy())
                val_trues.extend(calories.cpu().numpy())
                val_dish_ids.extend(dish_id)
        val_mae = mae_metric.compute()
        val_rmse = mse_metric.compute().sqrt()
        mae_metric.reset()
        mse_metric.reset()

        print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {total_loss/len(train_loader):.2f} | "
              f"Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'ingr_to_idx': ingr_to_idx,
                'config': config
            }, config.MODEL_SAVE_PATH)
            print(f"→ Saved best model (MAE: {val_mae:.2f})")

    # === СОХРАНЕНИЕ ИНФЕРЕНС-БАНДЛА (БЕЗ ЗАВИСИМОСТИ ОТ data/images/) ===
    print("Сохранение inference bundle (эмбеддинги + метаданные)...")
    model.eval()
    transforms_emb = get_transforms(config, is_train=False)
    emb_dataset = CalorieDataset(train_df, ingr_to_idx, config.IMAGES_DIR, transforms_emb, config.MAX_INGREDIENTS, mode="train")
    emb_loader = DataLoader(emb_dataset, batch_size=64, shuffle=False, num_workers=4)

    embeddings = []
    dish_ids = []
    calories = []
    masses = []
    ingredient_names_list = []

    with torch.no_grad():
        for image, ingr, cal, mass, did in tqdm(emb_loader, desc="Extracting train embeddings"):
            image, ingr = image.to(device), ingr.to(device)
            _, img_emb, text_emb = model(image, ingr)
            if config.IMAGE_UNFREEZE_LAYERS is not None:
                # multimodal
                fused = torch.cat([img_emb, text_emb], dim=1)
                emb = fused.cpu().numpy()
            else:
                # fallback
                fused = torch.cat([img_emb, text_emb], dim=1)
                emb = fused.cpu().numpy()
            embeddings.append(emb)
            dish_ids.extend(did)
            calories.extend(cal.numpy())
            masses.extend(mass.numpy())
            for i in range(len(did)):
                row = train_df[train_df["dish_id"] == did[i]].iloc[0]
                ingredient_names_list.append(row["ingredient_names"])

    embeddings = np.vstack(embeddings)

    bundle = {
        "embeddings": embeddings,
        "dish_ids": dish_ids,
        "calories": calories,
        "masses": masses,
        "ingredient_names": ingredient_names_list,
        "ingr_to_idx": ingr_to_idx,
        "config": config
    }
    torch.save(bundle, config.EMBEDDINGS_PATH)
    print(f"✅ Inference bundle saved to: {config.EMBEDDINGS_PATH}")

    return model, ingr_to_idx