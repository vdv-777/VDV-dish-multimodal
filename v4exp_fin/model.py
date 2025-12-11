import torch
import torch.nn as nn
import timm

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.bn(self.fc(self.relu(x))))

class CaloriePredictionModel(nn.Module):
    def __init__(self, cfg, num_ingredients):
        super().__init__()
        self.image_encoder = timm.create_model(cfg.IMAGE_MODEL_NAME, pretrained=True, num_classes=0)
        self.image_proj = nn.Linear(self.image_encoder.num_features, cfg.HIDDEN_DIM)
        self.ingredient_embedding = nn.Embedding(num_ingredients, cfg.EMBEDDING_DIM, padding_idx=0)
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.EMBEDDING_DIM * cfg.MAX_INGREDIENTS, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT)
        )
        self.mass_proj = nn.Linear(1, cfg.HIDDEN_DIM)
        
        self.mixer = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM * 3, cfg.HIDDEN_DIM),
            nn.BatchNorm1d(cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            ResidualBlock(cfg.HIDDEN_DIM, cfg.DROPOUT),
            nn.Linear(cfg.HIDDEN_DIM, 1)
        )

        self.freeze_image_encoder(cfg.IMAGE_UNFREEZE_LAYERS or [])

    def freeze_image_encoder(self, unfreeze_patterns):
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = any(name.startswith(p) for p in unfreeze_patterns)

    def forward(self, image, ingr_indices, mass):
        img_feat = self.image_encoder(image)
        img_emb = self.image_proj(img_feat)
        
        ingr_emb = self.ingredient_embedding(ingr_indices)
        ingr_emb = ingr_emb.view(ingr_emb.size(0), -1)
        text_emb = self.text_proj(ingr_emb)
        
        mass_emb = self.mass_proj(mass.unsqueeze(1))
        
        fused = torch.cat([img_emb, text_emb, mass_emb], dim=1)
        return self.mixer(fused).squeeze(-1)