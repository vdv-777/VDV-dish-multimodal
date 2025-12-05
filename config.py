from dataclasses import dataclass
import os
from typing import List, Optional

@dataclass
class Config:
    # Paths
    DATA_DIR: str = "data"
    INGREDIENTS_PATH: str = os.path.join(DATA_DIR, "ingredients.csv")
    DISH_PATH: str = os.path.join(DATA_DIR, "dish.csv")
    IMAGES_DIR: str = os.path.join(DATA_DIR, "images")
    
    # Model
    IMAGE_MODEL_NAME: str = "tf_efficientnet_b0"
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 256
    
    # Training
    BATCH_SIZE: int = 32
    EPOCHS: int = 30
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    SEED: int = 42
    
    # Data
    MAX_INGREDIENTS: int = 20
    DEBUG: bool = False
    DEBUG_SIZE: int = 500
    
    # Fine-tuning control (CRITICAL FOR SMALL DATASETS)
    IMAGE_UNFREEZE_LAYERS: Optional[List[str]] = None  # e.g., ["blocks.6", "conv_head", "bn2"]
    
    # IO
    SAVE_DIR: str = "runs/exp_f"
    MODEL_SAVE_PATH: str = os.path.join(SAVE_DIR, "best_model.pth")
    EMBEDDINGS_PATH: str = os.path.join(SAVE_DIR, "train_embeddings.pth")
    LOG_PATH: str = os.path.join(SAVE_DIR, "train.log") 