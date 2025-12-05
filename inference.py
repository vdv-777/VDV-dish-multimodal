# inference.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dataset import CalorieDataset, get_transforms

def run_inference(config, model, ingr_to_idx, inference_df, device):
    model.eval()
    transforms = get_transforms(config, is_train=False)
    dataset = CalorieDataset(inference_df, ingr_to_idx, "vdv-dish", transforms, config.MAX_INGREDIENTS, mode="inference")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Загрузка нужного типа эмбеддингов
    print(f"Загрузка inference bundle из: {config.EMBEDDINGS_PATH}")
    bundle = torch.load(config.EMBEDDINGS_PATH, map_location="cpu")
    
    if config.INFERENCE_MODE == "image_only":
        train_embs = bundle["image_embeddings"]
    else:
        train_embs = bundle["multimodal_embeddings"]

    train_dish_ids = bundle["dish_ids"]
    train_calories = bundle["calories"]
    train_masses = bundle["masses"]
    train_ingredient_names = bundle["ingredient_names"]

    results = []

    with torch.no_grad():
        for image, ingr, dish_numbers in tqdm(loader, desc="Inference"):
            image, ingr = image.to(device), ingr.to(device)
            _, img_emb, text_emb = model(image, ingr)
            
            if config.INFERENCE_MODE == "image_only":
                query_emb = img_emb.cpu().numpy()
            else:
                fused = torch.cat([img_emb, text_emb], dim=1)
                query_emb = fused.cpu().numpy()

            sims = cosine_similarity(query_emb, train_embs)
            best_idx = np.argmax(sims, axis=1)

            for i, dish_num in enumerate(dish_numbers):
                idx = best_idx[i]
                results.append({
                    "dish_number": dish_num.item(),
                    "matched_dish_id": train_dish_ids[idx],
                    "Вес": train_masses[idx],
                    "Калорийность": train_calories[idx],
                    "Ингредиенты": ", ".join(train_ingredient_names[idx])
                })

    return pd.DataFrame(results)