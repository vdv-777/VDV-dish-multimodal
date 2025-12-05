import torch
import torch.nn as nn
import timm

class MultimodalCalorieModel(nn.Module):
    def __init__(self, cfg, num_ingredients):
        super().__init__()
        # Image backbone
        self.image_encoder = timm.create_model(cfg.IMAGE_MODEL_NAME, pretrained=True, num_classes=0)
        
        # Freeze all layers by default
        self.freeze_image_encoder(unfreeze_patterns=cfg.IMAGE_UNFREEZE_LAYERS or [])
        
        # Projection heads
        self.image_proj = nn.Linear(self.image_encoder.num_features, cfg.HIDDEN_DIM)

        # Ingredient embedding (text modality)
        self.ingredient_embedding = nn.Embedding(num_ingredients, cfg.EMBEDDING_DIM, padding_idx=0)
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.EMBEDDING_DIM * cfg.MAX_INGREDIENTS, cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Fusion and regression
        self.fusion = nn.Linear(cfg.HIDDEN_DIM * 2, cfg.HIDDEN_DIM)
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.HIDDEN_DIM, 1)
        )

    def freeze_image_encoder(self, unfreeze_patterns):
        """
        Замораживает все параметры image_encoder, кроме слоёв,
        чьи имена начинаются с одного из префиксов в unfreeze_patterns.
        """
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in unfreeze_patterns)

    def forward(self, image, ingr_indices):
        # Image features
        img_feat = self.image_encoder(image)
        img_emb = self.image_proj(img_feat)

        # Ingredient features
        ingr_emb = self.ingredient_embedding(ingr_indices)  # [B, MAX_INGREDIENTS, EMBEDDING_DIM]
        ingr_emb = ingr_emb.view(ingr_emb.size(0), -1)      # [B, MAX_INGREDIENTS * EMBEDDING_DIM]
        text_emb = self.text_proj(ingr_emb)

        # Fusion
        fused = torch.cat([img_emb, text_emb], dim=1)       # [B, 2 * HIDDEN_DIM]
        fused = self.fusion(fused)                          # [B, HIDDEN_DIM]
        output = self.regressor(fused).squeeze(-1)          # [B]

        return output, img_emb, text_emb