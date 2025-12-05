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
    def __init__(self, df, ingr_to_idx, img_dir, transforms, max_ingredients=20, mode="train"):
        """
        Args:
            df: DataFrame с данными
            ingr_to_idx: dict, маппинг названия ингредиента → индекс
            img_dir: путь к папке с изображениями
            transforms: albumentations.Compose
            max_ingredients: максимальное число ингредиентов
            mode: "train" или "inference"
        """
        self.df = df.reset_index(drop=True)
        self.ingr_to_idx = ingr_to_idx
        self.img_dir = img_dir
        self.transforms = transforms
        self.max_ingredients = max_ingredients
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # === ЗАГРУЗКА ИЗОБРАЖЕНИЯ ===
        if self.mode == "train":
            # В обучающем датасете: dish_id → папка/{dish_id}/rgb.png
            dish_id = row["dish_id"]
            img_path = os.path.join(self.img_dir, str(dish_id), "rgb.png")
        else:
            # В инференсе: dish_number → файл {dish_number}.png в корне img_dir
            dish_number = row["dish_number"]
            img_path = os.path.join(self.img_dir, f"{dish_number}.png")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Заглушка при отсутствии изображения
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        image = np.array(image)
        image = self.transforms(image=image)["image"]

        # === ОБРАБОТКА ИНГРЕДИЕНТОВ ===
        if self.mode == "train":
            ingr_names = row.get("ingredient_names", [])
            if isinstance(ingr_names, str):
                ingr_names = ingr_names.split(",") if ingr_names else []
            indices = [self.ingr_to_idx.get(name, 0) for name in ingr_names]
            indices = indices[:self.max_ingredients]
            indices += [0] * (self.max_ingredients - len(indices))
            indices = torch.tensor(indices, dtype=torch.long)

            calories = torch.tensor(row["total_calories"], dtype=torch.float32)
            mass = torch.tensor(row["total_mass"], dtype=torch.float32)
            return image, indices, calories, mass, dish_id

        else:
            # Режим инференса
            ingr_names = row.get("ingredient_names", [])
            if isinstance(ingr_names, str):
                ingr_names = ingr_names.split(",") if ingr_names else []
            indices = [self.ingr_to_idx.get(name.strip(), 0) for name in ingr_names]
            indices = indices[:self.max_ingredients]
            indices += [0] * (self.max_ingredients - len(indices))
            indices = torch.tensor(indices, dtype=torch.long)
            dish_number = row["dish_number"]
            return image, indices, dish_number


def get_transforms(cfg, is_train=True):
    """
    Возвращает albumentations.Compose с трансформациями.
    Поддерживает albumentations >=2.0 (ToTensorV2 импортируется отдельно).
    """
    try:
        model_cfg = timm.get_pretrained_cfg(cfg.IMAGE_MODEL_NAME)
        size = model_cfg.input_size[1]
        mean, std = model_cfg.mean, model_cfg.std
    except:
        size = 224
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if is_train:
        return A.Compose([
            A.SmallestMaxSize(max_size=int(size * 1.2)),
            A.RandomCrop(size, size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.8),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])