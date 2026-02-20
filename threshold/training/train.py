from multiprocessing import reduction
import torch
import torch.nn as nn
from threshold.utils.compute_recall_threshold import compute_recall_threshold
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


class EntroMixLoss(nn.Module):
    """
    Dynamic Divergence Gate Loss for mixed real and synthetic data training.
    
    - Real Data: Always contributes to loss.
    - Syn Data: Only contributes if JSD <= pacing_threshold.
    
    This implements curriculum learning by gradually incorporating synthetic samples
    based on their learnability (JSD) score.
    """
    def __init__(self, device):
        super().__init__()
        class_weights = torch.tensor([803/1569, 1569/803], dtype=torch.float32, device=device)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')  # Per-sample loss

    def forward(self, logits, targets, is_syn, jsd_scores, pacing_threshold):
        """
        Apply the Dynamic Divergence Gate.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            is_syn: Binary flags indicating synthetic samples [batch_size]
            jsd_scores: JSD divergence scores for each sample [batch_size]
            pacing_threshold: Current threshold for synthetic sample acceptance
        
        Returns:
            Scalar loss value
        """
        # 1. Compute standard Cross Entropy for everyone
        raw_losses = self.ce_loss(logits, targets)
        
        # 2. Compute the Gate Mask (Phi function)
        # Real data always passes (gate = 1)
        real_mask = (~is_syn).float()
        
        # Syn data passes only if JSD is low enough for current epoch
        # "Learnability Score" check
        syn_passing = (jsd_scores <= pacing_threshold).float()
        syn_mask = is_syn.float() * syn_passing
        
        # Combine masks
        final_mask = real_mask + syn_mask
        
        # 3. Apply Mask
        masked_loss = raw_losses * final_mask
        
        # 4. Average over the number of *valid* samples (avoid divide by zero)
        num_valid = final_mask.sum()
        if num_valid == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        return masked_loss.sum() / num_valid


def train_model_threshold(
    model,
    data_dir,
    val_loader,
    test_loader,
    optimizer,
    device,
    epochs,
    batch_size,
    get_train_loader,
    schedule=None,
    scheduler=None,
    dataset_csv='dataset.csv',
    warmup_epochs=10,
    jsd_start=0.1,
    jsd_end=1.0,
    num_workers=0,
    start_epoch=0,
    checkpoint_dir=None,
    save_every=1,
    history=None,
    use_entromix=True
):
    """
    Train model with threshold-based curriculum learning using EntroMixLoss.
        num_workers=0,
        warmup_epochs=10,
    Args:
        model: Neural network model
        data_dir: Data directory
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        optimizer: Optimizer
        device: Device (cuda/cpu)
        epochs: Number of epochs
        batch_size: Batch size
        get_train_loader: Function to get train loader
        schedule: Optional legacy schedule (ignored when using warmup/jsd ramp)
        scheduler: Learning rate scheduler
        use_entromix: Whether to use EntroMixLoss (default: True)
        warmup_epochs: Epochs with real-only data (no synthetic)
        jsd_start: JSD threshold to start admitting synthetic after warmup
        jsd_end: JSD threshold reached by the final epoch
        num_workers: Number of DataLoader workers
        start_epoch: Epoch index to start from when resuming
        checkpoint_dir: Directory to save checkpoints (None to disable)
        save_every: Save checkpoint every N epochs
        history: Optional history dict to continue plots/metrics
    
    Returns:
            num_workers: Number of DataLoader workers
        Dictionary with training history and final metrics
    """
    
    # Loss function
    if use_entromix:
        criterion = EntroMixLoss(device)
        print("[OK] Using EntroMixLoss with Dynamic Divergence Gate")
    else:
        class_weights = torch.tensor([803/1569, 1569/803]).to(device)  # [0.51, 1.95]
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("[OK] Using Standard CrossEntropyLoss")
    
    # Standard criterion for validation (always use standard loss for val/test)
    # class_weights = torch.tensor([803/1569, 1569/803]).to(device)
    val_criterion = nn.CrossEntropyLoss()
    
    train_losses = list(history.get('train_losses', [])) if history else []
    val_losses = list(history.get('val_losses', [])) if history else []
    train_recalls = list(history.get('train_recalls', [])) if history else []
    val_recalls = list(history.get('val_recalls', [])) if history else []
    train_losses_detailed = list(history.get('train_losses_detailed', [])) if history else []
    
    all_val_preds = []
    all_val_labels = []
    
    def compute_epoch_threshold(epoch_idx: int):
        """Real-only during warmup, then linear ramp of JSD threshold."""
        if epoch_idx < warmup_epochs:
            return None  # no synthetic in loader
        progress = (epoch_idx - warmup_epochs) / max(1, (epochs - warmup_epochs))
        progress = max(0.0, min(1.0, progress))
        return jsd_start + progress * (jsd_end - jsd_start)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        # ---- Select threshold for this epoch ----
        current_threshold = compute_epoch_threshold(epoch)
        
        train_loader = get_train_loader(
            data_dir=data_dir,
            batch_size=batch_size,
            threshold=current_threshold,
              dataset_csv=dataset_csv,
              num_workers=num_workers
        )
        
        phase_name = "Real only" if current_threshold is None else f"Threshold: {current_threshold:.4f}"
        print(f"\nEpoch {epoch+1}/{epochs} | {phase_name}")
        
        # -------- TRAINING --------
        model.train()
        total_loss = 0
        num_batches = 0
        
        for imgs, labels, is_syn, jsd_scores in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            is_syn = is_syn.to(device)
            jsd_scores = jsd_scores.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            # Use EntroMixLoss with dynamic divergence gate
            if isinstance(criterion, EntroMixLoss):
                loss = criterion(outputs, labels, is_syn, jsd_scores, current_threshold if current_threshold is not None else 1.0)
            else:
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)
        train_losses_detailed.append({
            'epoch': epoch + 1,
            'threshold': current_threshold,
            'loss': avg_train_loss
        })
        
        # -------- EVALUATE RECALLS --------
        train_recall = compute_recall_threshold(model, train_loader, device, threshold=0.5)
        val_recall = compute_recall_threshold(model, val_loader, device, threshold=0.3)
        
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        
        print(f"Train Recall: {train_recall:.3f} | Val Recall: {val_recall:.3f}")
        
        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                # Use standard loss for validation (no EntroMixLoss)
                val_loss += val_criterion(outputs, labels).item()
                
                # Collect predictions
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        all_val_preds.extend(val_preds)
        all_val_labels.extend(val_labels_list)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        if checkpoint_dir and save_every > 0:
            if ((epoch + 1) % save_every == 0) or ((epoch + 1) == epochs):
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "history": {
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "train_recalls": train_recalls,
                        "val_recalls": val_recalls,
                        "train_losses_detailed": train_losses_detailed
                    }
                }
                last_path = os.path.join(checkpoint_dir, "last.pth")
                torch.save(checkpoint, last_path)
                print(f"[OK] Checkpoint saved: {last_path}")
    
    # -------- FINAL METRICS ON TEST SET --------
    os.makedirs("outputs/plots", exist_ok=True)
    
    # Evaluate on test set
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            
            # Collect predictions
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/plots/loss_curve.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Loss curve saved to: outputs/plots/loss_curve.png")
    plt.close()
    
    # Plot recall curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_recalls, label='Train Recall', marker='o')
    plt.plot(range(1, epochs + 1), val_recalls, label='Val Recall', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training and Validation Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/plots/recall_curve.png', dpi=300, bbox_inches='tight')
    print("[OK] Recall curve saved to: outputs/plots/recall_curve.png")
    plt.close()
    
    # Compute confusion matrix on TEST SET
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('outputs/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("[OK] Confusion matrix saved to: outputs/plots/confusion_matrix.png")
    plt.close()
    
    # Compute detailed metrics
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # Recall of malignant (class 1)
    specificity = tn / (tn + fp)  # Recall of benign (class 0)
    f1 = f1_score(test_labels, test_preds)
    recall_malignant = sensitivity  # TP / (TP + FN)
    
    # Print final metrics
    print("\n" + "="*70)
    print("FINAL METRICS (Test Set)")
    print("="*70)
    print(f"Overall Accuracy:        {accuracy:.4f}")
    print(f"Recall (Malignant):      {recall_malignant:.4f}")
    print(f"Sensitivity (Malignant): {sensitivity:.4f}")
    print(f"Specificity (Benign):    {specificity:.4f}")
    print(f"F1 Score:                {f1:.4f}")
    print("="*70)
    
    # Save metrics to file
    with open('outputs/final_metrics.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("FINAL METRICS (Test Set)\n")
        f.write("="*70 + "\n")
        f.write(f"Overall Accuracy:        {accuracy:.4f}\n")
        f.write(f"Recall (Malignant):      {recall_malignant:.4f}\n")
        f.write(f"Sensitivity (Malignant): {sensitivity:.4f}\n")
        f.write(f"Specificity (Benign):    {specificity:.4f}\n")
        f.write(f"F1 Score:                {f1:.4f}\n")
        f.write("="*70 + "\n")
        f.write("\nEpoch-wise Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Epoch':<8} {'Threshold':<12} {'Train Loss':<15} {'Train Recall':<15}\n")
        f.write("-"*70 + "\n")
        for epoch in range(len(train_losses_detailed)):
            info = train_losses_detailed[epoch]
            tr = train_recalls[epoch]
            f.write(f"{info['epoch']:<8} {str(info['threshold']):<12} "
                   f"{info['loss']:<15.4f} {tr:<15.4f}\n")
    
    print("[OK] Final metrics saved to: outputs/final_metrics.txt\n")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_recalls': train_recalls,
        'val_recalls': val_recalls,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'recall_malignant': recall_malignant,
        'confusion_matrix': cm
    }
