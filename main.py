import sys
import os
import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Threshold-based curriculum learning training"
    )
    parser.add_argument("--data-dir", default="data", help="Dataset root directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=140, help="Number of epochs")
    parser.add_argument("--dataset-csv", default="dataset.csv", help="Dataset CSV file")
    parser.add_argument(
        "--model",
        dest="model_choice",
        default="resnet50",
        choices=["resnet50", "vit", "swin"],
        help="Model choice",
    )
    parser.add_argument(
        "--device",
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device override, e.g. 'cpu' or 'cuda'. Default: auto",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Epochs with real-only data",
    )
    parser.add_argument(
        "--jsd-start",
        type=float,
        default=0.03,
        help="Starting JSD threshold for synthetic data",
    )
    parser.add_argument(
        "--jsd-end",
        type=float,
        default=1.0,
        help="Ending JSD threshold for synthetic data",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint (.pth) to resume from",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="outputs/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--use-entromix",
        dest="use_entromix",
        action="store_true",
        default=True,
        help="Use EntroMixLoss (default: enabled)",
    )
    parser.add_argument(
        "--no-entromix",
        dest="use_entromix",
        action="store_false",
        help="Disable EntroMixLoss and use standard CrossEntropyLoss",
    )
    return parser.parse_args()


def main(args):
    """
    Main training script with threshold-based curriculum learning.
    """

    # ============= CONFIG =============
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = args.data_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    epochs = args.epochs
    dataset_csv = args.dataset_csv
    model_choice = args.model_choice
    
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
    warmup_epochs = args.warmup_epochs     # Phase 1: real-only
    jsd_threshold_start = args.jsd_start   # Start admitting very clean synthetic
    jsd_threshold_end = args.jsd_end       # End by admitting all synthetic
    
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,eta_min=1e-6)

    # ============= RESUME (OPTIONAL) =============
    start_epoch = 0
    history = None
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])

            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(device)

            if "scheduler_state_dict" in checkpoint and scheduler is not None:
                scheduler_state = checkpoint.get("scheduler_state_dict")
                if scheduler_state:
                    scheduler.load_state_dict(scheduler_state)

            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            history = checkpoint.get("history")
            print(f"[OK] Resuming from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("[OK] Loaded model weights only (no optimizer/scheduler state)")
    
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
        start_epoch=start_epoch,
        batch_size=batch_size,
        schedule=None,
        warmup_epochs=warmup_epochs,
        jsd_start=jsd_threshold_start,
        jsd_end=jsd_threshold_end,
        get_train_loader=get_train_loader_threshold,
        scheduler=scheduler,
        dataset_csv=dataset_csv,
        num_workers=num_workers,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        history=history,
        use_entromix=args.use_entromix
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
    main(parse_args())
