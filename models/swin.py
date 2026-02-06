import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SwinTransformer(nn.Module):
    """
    Classifier using Swin Transformer (Swin-B).
    This module uses a pretrained Swin Transformer as the backbone
    and adds a custom classifier head on top. Backbones are frozen
    by default for efficient training.
    """
    def __init__(self, num_classes: int = 2, freeze_backbones: bool = True):
        super().__init__()

        # Swin Transformer-B feature extractor
        swin = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        
        # Remove pretrained classification head so the model returns
        # feature embeddings instead of 1000-way logits
        if hasattr(swin, 'head'):
            swin.head = nn.Identity()
        elif hasattr(swin, 'classifier'):
            swin.classifier = nn.Identity()
        
        self.swin = swin
        swin_out = 1024  # Swin-B hidden dimension (feature size)

        if freeze_backbones:
            for p in self.swin.parameters():
                p.requires_grad = False

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(swin_out, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(384, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features from Swin Transformer
        feat = self.swin(x)
        
        # Handle different output formats
        if feat.dim() == 3:
            # If output is (B, num_tokens, hidden_dim), take CLS token
            feat = feat[:, 0, :]
        elif feat.dim() == 2:
            # Already in (B, hidden_dim) format
            feat = feat
        else:
            # Flatten if needed
            feat = torch.flatten(feat, 1)

        # Pass through classifier
        logits = self.classifier(feat)
        return logits


def get_swin(num_classes: int = 2) -> nn.Module:
    """Factory for Swin Transformer with frozen backbones."""
    return SwinTransformer(num_classes=num_classes, freeze_backbones=True)
