import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Import models from models folder
from models.vit import ViT
from models.swin import SwinTransformer
from models.resnet50 import get_resnet50

from threshold.utils.dataloader import get_train_loader_threshold, get_val_loader
from threshold.training.train import train_model_threshold


"""
EntroMixLoss Integration:
========================

This script uses EntroMixLoss (Dynamic Divergence Gate) for curriculum learning
with mixed real and synthetic data.

How it works:
- Real data always contributes to loss (gate = 1)
- Synthetic data only contributes if JSD score <= pacing_threshold
- Threshold increases gradually with epochs (curriculum learning)
- Allows gradual incorporation of synthetic samples as model matures

The training schedule in main() controls when synthetic data enters training:
  Epochs 0-10: Only real images (threshold = None)
  Epochs 10-20: Include synthetic with JSD <= 0.00000005
  Epochs 20-30: Include synthetic with JSD <= 0.0000005
  Epochs 30-40: Include all synthetic samples (threshold = 1.0)

To disable EntroMixLoss and use standard CrossEntropyLoss, edit
threshold/training/train.py and set: use_entromix = False
"""


def main():
    """
    Main training script with threshold-based curriculum learning.
    """
    
    # ============= CONFIG =============
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = 'data'
    batch_size = 32
    num_workers = 0
    learning_rate = 0.003
    epochs = 20
    dataset_csv = 'dataset.csv'
    model_choice = 'resnet50'  # Options: 'resnet50', 'vit', 'swin', 'efficientnet_resnet', 'dual_backbone'
    
    print("="*70)
    print("THRESHOLD-BASED CURRICULUM LEARNING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Num Workers: {num_workers}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Model: {model_choice}\n")
    
    # ============= CURRICULUM PACE =============
    warmup_epochs = 15              # Phase 1: real-only
    jsd_threshold_start = 0.2       # Start admitting very clean synthetic
    jsd_threshold_end = 0.8        # End by admitting all synthetic
    
    print("Curriculum:")
    print(f"  Epochs 0-{warmup_epochs-1}: real-only (no synthetic)")
    print(f"  Epochs {warmup_epochs}-{epochs-1}: linear ramp JSD {jsd_threshold_start} -> {jsd_threshold_end}")
    print()
    
    # ============= MODEL =============
    print("Loading model...")
    if model_choice == 'resnet50':
        model = get_resnet50(num_classes=2)
        print(f"[OK] Model loaded: ResNet50")
    
    elif model_choice == 'vit':
        model = ViT(num_classes=2, freeze_backbones=False)
        print(f"[OK] Model loaded: Vision Transformer (ViT)")
    
    elif model_choice == 'swin':
        model = SwinTransformer(num_classes=2, freeze_backbones=False)
        print(f"[OK] Model loaded: Swin Transformer")
    
    elif model_choice == 'efficientnet_resnet':
        model = EfficientNetResNetFusion(num_classes=2, freeze_backbones=False)
        print(f"[OK] Model loaded: EfficientNet + ResNet Fusion")
    else:
        raise ValueError(f"Unknown model: {model_choice}")
    
    model.to(device)
    print()
    
    # ============= OPTIMIZER & SCHEDULER =============
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # ============= DATA LOADERS =============
    val_loader = get_val_loader(data_dir, batch_size, num_workers=num_workers)
    print(f"[OK] Validation loader ready")
    
    # Get test loader
    from threshold.utils.dataloader import get_test_loader
    test_loader = get_test_loader(data_dir, batch_size, num_workers=num_workers)
    print(f"[OK] Test loader ready\n")
    
    # ============= TRAINING =============
    results = train_model_threshold(
        model=model,
        data_dir=data_dir,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        schedule=None,
        warmup_epochs=warmup_epochs,
        jsd_start=jsd_threshold_start,
        jsd_end=jsd_threshold_end,
        get_train_loader=get_train_loader_threshold,
        scheduler=scheduler,
        dataset_csv=dataset_csv,
        num_workers=num_workers
    )
    
    # ============= SAVE MODEL =============
    os.makedirs('outputs', exist_ok=True)
    model_path = f'outputs/threshold_trained_{model_choice}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Model saved to: {model_path}\n")
    
    print("="*70)
    print("[OK] TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
