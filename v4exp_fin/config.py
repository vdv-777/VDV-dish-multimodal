from dataclasses import dataclass, field
import os
from typing import List, Optional

@dataclass
class Config:
    # Пути
    DATA_DIR: str = "data"
    INGREDIENTS_PATH: str = os.path.join(DATA_DIR, "ingredients.csv")
    DISH_PATH: str = os.path.join(DATA_DIR, "dish.csv")
    IMAGES_DIR: str = os.path.join(DATA_DIR, "images")
    
    # Режим
    TRAIN_MODE: str = "multimodal_with_mass"
    
    # Модель
    IMAGE_MODEL_NAME: str = "tf_efficientnet_b3"
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 256
    
    # Обучение
    BATCH_SIZE: int = 16
    EPOCHS: int = 35
    IMAGE_LR: float = 7e-5
    HEAD_LR: float = 2e-3
    WEIGHT_DECAY: float = 3e-5
    DROPOUT: float = 0.4
    SEED: int = 42
    
    # Fine-tuning
    IMAGE_UNFREEZE_LAYERS: Optional[List[str]] = field(
        default_factory=lambda: ["blocks.6", "blocks.7", "conv_head", "bn2"]
    )
    
    # Данные
    MAX_INGREDIENTS: int = 20
    DEBUG: bool = False
    DEBUG_SIZE: int = 200
    
    # Вывод
    SAVE_DIR: str = "runs/exp_multimodal_mass_b3"
    MODEL_SAVE_PATH: str = os.path.join(SAVE_DIR, "best_model.pth")