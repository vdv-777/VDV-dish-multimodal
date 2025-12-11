import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import torch

class CalorieDataset(Dataset):
    def __init__(self, df, ingr_to_idx, img_dir, transforms, max_ingredients=20, mode="train", config=None):
        self.df = df.reset_index(drop=True)
        self.ingr_to_idx = ingr_to_idx
        self.img_dir = img_dir
        self.transforms = transforms
        self.max_ingredients = max_ingredients
        self.mode = mode
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # === ЗАГРУЗКА ИЗОБРАЖЕНИЯ ===
        if self.mode == "train":
            dish_id = row["dish_id"]
            img_path = os.path.join(self.img_dir, str(dish_id), "rgb.png")
        else:
            # Инференс: изображение = {dish_number}.png
            dish_number = row["dish_number"]
            img_path = os.path.join(self.img_dir, f"{dish_number}.png")

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (300, 300), (0, 0, 0))
        image = np.array(image)
        image = self.transforms(image=image)["image"]

        if self.mode == "train":
            # Ингредиенты
            ingr_names = row.get("ingredient_names", [])
            if isinstance(ingr_names, str):
                ingr_names = ingr_names.split(",") if ingr_names else []
            indices = [self.ingr_to_idx.get(name, 0) for name in ingr_names]
            indices = indices[:self.max_ingredients]
            indices += [0] * (self.max_ingredients - len(indices))
            indices = torch.tensor(indices, dtype=torch.long)
            
            mass = torch.tensor(row["total_mass"], dtype=torch.float32)
            calories = torch.tensor(row["total_calories"], dtype=torch.float32)
            return image, indices, mass, calories
        
        else:
            # Инференс
            ingr_names = row.get("ingredient_names", [])
            if isinstance(ingr_names, str):
                ingr_names = ingr_names.split(",") if ingr_names else []
            indices = [self.ingr_to_idx.get(name.strip(), 0) for name in ingr_names]
            indices = indices[:self.max_ingredients]
            indices += [0] * (self.max_ingredients - len(indices))
            indices = torch.tensor(indices, dtype=torch.long)
            
            dish_number = row["dish_number"]
            # Используем медианную массу, если колонка 'mass' отсутствует
            mass_val = row.get("mass", 218.0)  # 218.0 = медиана из данных
            mass = torch.tensor(mass_val, dtype=torch.float32)
            return image, indices, mass, dish_number


def get_transforms(config, is_train=True):
    try:
        model_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
        size = model_cfg.input_size[1]
        mean, std = model_cfg.mean, model_cfg.std
    except:
        size = 300
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if is_train:
        return A.Compose([
            A.SmallestMaxSize(max_size=int(size * 1.2)),
            A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), translate_percent=(-0.1, 0.1),
                     shear=(-10, 10), p=0.8),
            A.CoarseDropout(num_holes_range=(2, 8),
                            hole_height_range=(int(0.07*size), int(0.15*size)),
                            hole_width_range=(int(0.1*size), int(0.15*size)), p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.7),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])