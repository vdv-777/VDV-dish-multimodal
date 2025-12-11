import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError
import pandas as pd
from tqdm import tqdm
from dataset import CalorieDataset, get_transforms
from model import CaloriePredictionModel

def prepare_ingredient_mapping(ingredients_df, dish_df):
    ingr_id_to_name = dict(zip(ingredients_df["id"], ingredients_df["ingr"]))
    def id_list_to_names(ingr_str):
        if pd.isna(ingr_str): return []
        ids = ingr_str.split(";")
        names = []
        for iid in ids:
            if iid.startswith("ingr_"):
                try:
                    num = int(iid.replace("ingr_", ""))
                    name = ingr_id_to_name.get(num)
                    if name: names.append(name)
                except: continue
        return names
    dish_df = dish_df.copy()
    dish_df["ingredient_names"] = dish_df["ingredients"].apply(id_list_to_names)
    return dish_df

def train(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    ingredients_df = pd.read_csv(config.INGREDIENTS_PATH)
    dish_df = pd.read_csv(config.DISH_PATH)
    dish_df = prepare_ingredient_mapping(ingredients_df, dish_df)

    if config.DEBUG:
        dish_df = dish_df.sample(n=min(config.DEBUG_SIZE, len(dish_df)), random_state=config.SEED).reset_index(drop=True)

    train_df = dish_df[dish_df["split"] == "train"]
    val_df = dish_df[dish_df["split"] == "test"]

    all_names = set()
    for names in dish_df["ingredient_names"]:
        all_names.update(names)
    ingr_to_idx = {name: idx + 1 for idx, name in enumerate(sorted(all_names))}
    ingr_to_idx["<PAD>"] = 0

    transforms_train = get_transforms(config, is_train=True)
    transforms_val = get_transforms(config, is_train=False)

    train_dataset = CalorieDataset(train_df, ingr_to_idx, config.IMAGES_DIR, transforms_train, config.MAX_INGREDIENTS, mode="train", config=config)
    val_dataset = CalorieDataset(val_df, ingr_to_idx, config.IMAGES_DIR, transforms_val, config.MAX_INGREDIENTS, mode="train", config=config)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CaloriePredictionModel(config, len(ingr_to_idx)).to(device)
    model.freeze_image_encoder(config.IMAGE_UNFREEZE_LAYERS or [])
    
    optimizer = torch.optim.AdamW([
        {'params': model.image_encoder.parameters(), 'lr': config.IMAGE_LR},
        {'params': model.image_proj.parameters(), 'lr': config.HEAD_LR},
        {'params': model.ingredient_embedding.parameters(), 'lr': config.HEAD_LR},
        {'params': model.text_proj.parameters(), 'lr': config.HEAD_LR},
        {'params': model.mass_proj.parameters(), 'lr': config.HEAD_LR},
        {'params': model.mixer.parameters(), 'lr': config.HEAD_LR},
    ], weight_decay=config.WEIGHT_DECAY)
    
    criterion = nn.HuberLoss(delta=1.0)
    mae = MeanAbsoluteError().to(device)
    best_val_mae = float('inf')

    for epoch in range(config.EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            image, ingr, mass, calories = batch
            image, ingr, mass, calories = image.to(device), ingr.to(device), mass.to(device), calories.to(device)
            pred = model(image, ingr, mass)
            loss = criterion(pred, calories)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Валидация
        model.eval()
        with torch.no_grad():
            for image, ingr, mass, calories in val_loader:
                image, ingr, mass, calories = image.to(device), ingr.to(device), mass.to(device), calories.to(device)
                pred = model(image, ingr, mass)
                mae.update(pred, calories)
        val_mae = mae.compute()
        mae.reset()
        print(f"Epoch {epoch+1} | Val MAE: {val_mae:.2f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"→ Saved best model (MAE: {val_mae:.2f})")

    return model, ingr_to_idx