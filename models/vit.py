import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ViT(nn.Module):
    """
    Classifier using only Vision Transformer (ViT-B/16).
    This module replaces the previous ViT backbone and exposes a
    simple classifier head on top of a pretrained ViT. Backbones
    are frozen by default.
    """
    def __init__(self, num_classes: int = 2, freeze_backbones: bool = True):
        super().__init__()

        # Vision Transformer (ViT-B/16) feature extractor
        vit = models.vit_b_16(pretrained=True)
        # Remove pretrained classification head so the model returns
        # feature embeddings (hidden dim) instead of 1000-way logits.
        def _remove_vit_head(m: nn.Module) -> bool:
            candidates = ["heads", "head", "classifier"]
            for name in candidates:
                # support nested attributes like 'heads.head'
                if "." in name:
                    obj = m
                    parts = name.split(".")
                    parent = None
                    ok = True
                    for p in parts[:-1]:
                        if hasattr(obj, p):
                            parent = obj
                            obj = getattr(obj, p)
                        else:
                            ok = False
                            break
                    if ok and hasattr(obj, parts[-1]):
                        setattr(obj, parts[-1], nn.Identity())
                        return True
                else:
                    if hasattr(m, name):
                        setattr(m, name, nn.Identity())
                        return True
            return False

        _remove_vit_head(vit)
        self.vit = vit
        vit_out = 768  # ViT hidden dimension (feature size)

        if freeze_backbones:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(vit_out, 768),
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
        # ViT forward may return either class logits (B, num_classes)
        # or token embeddings (B, num_tokens, hidden_dim) depending on
        # the torchvision implementation. Handle both cases.
        v = self.vit(x)

        if v.dim() == 3:
            feat = v[:, 0, :]
        elif v.dim() == 2:
            feat = v
        else:
            feat = torch.flatten(v, 1)

        logits = self.classifier(feat)
        return logits


def get_vit(num_classes: int = 2) -> nn.Module:
    """Factory for ViT with frozen backbones."""
    return ViT(num_classes=num_classes, freeze_backbones=True)