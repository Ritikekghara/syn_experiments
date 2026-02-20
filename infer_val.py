import argparse
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torchvision import datasets, transforms

from models.resnet50 import get_resnet50


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a validation folder with two classes."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/threshold_trained_resnet50_model.pth",
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--val-dir",
        default="data/val",
        help="Path to validation folder (must contain benign/ and malignant/)",
    )
    return parser.parse_args()


def load_model(checkpoint_path, device):
    model = get_resnet50(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_val_loader(val_dir, batch_size=32):
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_ds = datasets.ImageFolder(root=val_dir, transform=val_tf)
    if val_ds.classes != ["benign", "malignant"]:
        raise ValueError(
            f"Expected classes ['benign', 'malignant'], got {val_ds.classes}"
        )

    return torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def evaluate(model, loader, device):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(all_labels, all_preds)
    recall_malignant = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_benign = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "recall_malignant": recall_malignant,
        "sensitivity": recall_malignant,
        "specificity_benign": specificity_benign,
        "f1_score": f1,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.val_dir):
        raise NotADirectoryError(f"Validation folder not found: {args.val_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.checkpoint, device)
    val_loader = get_val_loader(args.val_dir)
    results = evaluate(model, val_loader, device)

    print("=" * 70)
    print("VALIDATION INFERENCE")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Val Dir:    {args.val_dir}")
    print(f"Device:     {device}")
    print("=" * 70)
    print(f"Overall Accuracy:        {results['accuracy']:.4f}")
    print(f"Recall (Malignant):      {results['recall_malignant']:.4f}")
    print(f"Sensitivity (Malignant): {results['sensitivity']:.4f}")
    print(f"Specificity (Benign):    {results['specificity_benign']:.4f}")
    print(f"F1 Score:                {results['f1_score']:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(results["confusion_matrix"])
    print(f"TN: {results['tn']}  FP: {results['fp']}")
    print(f"FN: {results['fn']}  TP: {results['tp']}")


if __name__ == "__main__":
    main()
