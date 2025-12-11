import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CalorieDataset, get_transforms

def run_inference(config, model, ingr_to_idx, inference_df, device):
    model.eval()
    transforms = get_transforms(config, is_train=False)
    dataset = CalorieDataset(inference_df, ingr_to_idx, "vdv-dish", transforms, config.MAX_INGREDIENTS, mode="inference", config=config)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    results = []
    with torch.no_grad():
        for image, ingr, mass, dish_numbers in tqdm(loader, desc="Inference"):
            image, ingr, mass = image.to(device), ingr.to(device), mass.to(device)
            pred = model(image, ingr, mass)
            for i, dish_num in enumerate(dish_numbers):
                results.append({
                    "dish_number": dish_num.item(),
                    "Масса": mass[i].item(),
                    "Калорийность": pred[i].item()
                })
    return pd.DataFrame(results)