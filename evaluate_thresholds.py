"""
Evaluate model performance across different classification thresholds.
Computes recall (malignant), specificity (benign), accuracy, and confusion matrix.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from models.resnet50 import get_resnet50
from threshold.utils.dataloader import get_test_loader
import os


def evaluate_at_threshold(model, test_loader, device, threshold=0.5):
    """
    Evaluate model at a specific threshold.
    
    Returns:
        dict with metrics: accuracy, recall_malignant, specificity_benign, 
                          f1_score, confusion_matrix, predictions, labels
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of malignant (class 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Apply threshold
    predictions = (all_probs >= threshold).astype(int)
    
    # Compute metrics
    cm = confusion_matrix(all_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(all_labels, predictions)
    recall_malignant = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
    specificity_benign = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    f1 = f1_score(all_labels, predictions)
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'recall_malignant': recall_malignant,
        'sensitivity': recall_malignant,
        'specificity_benign': specificity_benign,
        'f1_score': f1,
        'confusion_matrix': cm,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'predictions': predictions,
        'labels': all_labels,
        'probabilities': all_probs
    }


def plot_confusion_matrix(cm, threshold, save_path=None):
    """Plot and optionally save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign (0)', 'Malignant (1)'],
                yticklabels=['Benign (0)', 'Malignant (1)'])
    plt.title(f'Confusion Matrix (Threshold = {threshold:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_metrics_vs_threshold(results, save_dir='outputs/threshold_analysis'):
    """Plot metrics as a function of threshold."""
    os.makedirs(save_dir, exist_ok=True)
    
    thresholds = [r['threshold'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    recalls = [r['recall_malignant'] for r in results]
    specificities = [r['specificity_benign'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(thresholds, accuracies, marker='o', label='Accuracy', linewidth=2)
    plt.plot(thresholds, recalls, marker='s', label='Recall (Malignant)', linewidth=2)
    plt.plot(thresholds, specificities, marker='^', label='Specificity (Benign)', linewidth=2)
    plt.plot(thresholds, f1_scores, marker='d', label='F1 Score', linewidth=2)
    
    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Performance Metrics vs Classification Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)
    plt.ylim(0, 1.05)
    
    save_path = os.path.join(save_dir, 'metrics_vs_threshold.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Metrics plot saved to: {save_path}")
    plt.close()


def main():
    """Main evaluation script."""
    
    # ============= CONFIG =============
    model_path = r'C:\Users\RITIK\Desktop\medical_classification\outputs\threshold_trained_resnet50_model.pth'
    data_dir = 'data'
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Thresholds to test
    thresholds = [0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    print("="*80)
    print("THRESHOLD EVALUATION SCRIPT")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Thresholds to test: {thresholds}")
    print("="*80 + "\n")
    
    # ============= LOAD MODEL =============
    print("Loading model...")
    model = get_resnet50(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("[OK] Model loaded successfully\n")
    
    # ============= LOAD TEST DATA =============
    print("Loading test data...")
    test_loader = get_test_loader(data_dir, batch_size)
    print(f"[OK] Test loader ready\n")
    
    # ============= EVALUATE AT EACH THRESHOLD =============
    results = []
    
    for threshold in thresholds:
        print(f"\n{'='*80}")
        print(f"Evaluating at Threshold = {threshold:.2f}")
        print(f"{'='*80}")
        
        result = evaluate_at_threshold(model, test_loader, device, threshold)
        results.append(result)
        
        # Print metrics
        print(f"\nMetrics:")
        print(f"  Overall Accuracy:        {result['accuracy']:.4f}")
        print(f"  Recall (Malignant):      {result['recall_malignant']:.4f}")
        print(f"  Sensitivity (Malignant): {result['sensitivity']:.4f}")
        print(f"  Specificity (Benign):    {result['specificity_benign']:.4f}")
        print(f"  F1 Score:                {result['f1_score']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {result['tn']}  FP: {result['fp']}")
        print(f"  FN: {result['fn']}  TP: {result['tp']}")
        
        # Save confusion matrix for this threshold
        os.makedirs('outputs/threshold_analysis', exist_ok=True)
        cm_path = f'outputs/threshold_analysis/confusion_matrix_th{threshold:.2f}.png'
        plot_confusion_matrix(result['confusion_matrix'], threshold, cm_path)
    
    # ============= SUMMARY TABLE =============
    print("\n" + "="*80)
    print("SUMMARY: ALL THRESHOLDS")
    print("="*80)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Recall(Mal)':<15} {'Spec(Ben)':<15} {'F1':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['threshold']:<12.2f} {r['accuracy']:<12.4f} {r['recall_malignant']:<15.4f} "
              f"{r['specificity_benign']:<15.4f} {r['f1_score']:<10.4f}")
    
    # ============= FIND BEST THRESHOLDS =============
    print("\n" + "="*80)
    print("BEST THRESHOLDS BY METRIC")
    print("="*80)
    
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    best_recall = max(results, key=lambda x: x['recall_malignant'])
    best_f1 = max(results, key=lambda x: x['f1_score'])
    best_balanced = max(results, key=lambda x: x['recall_malignant'] + x['specificity_benign'])
    
    print(f"Best Accuracy:     Threshold = {best_accuracy['threshold']:.2f}, "
          f"Accuracy = {best_accuracy['accuracy']:.4f}")
    print(f"Best Recall (Mal): Threshold = {best_recall['threshold']:.2f}, "
          f"Recall = {best_recall['recall_malignant']:.4f}")
    print(f"Best F1 Score:     Threshold = {best_f1['threshold']:.2f}, "
          f"F1 = {best_f1['f1_score']:.4f}")
    print(f"Best Balanced:     Threshold = {best_balanced['threshold']:.2f}, "
          f"Recall+Spec = {best_balanced['recall_malignant'] + best_balanced['specificity_benign']:.4f}")
    
    # ============= SAVE RESULTS =============
    os.makedirs('outputs/threshold_analysis', exist_ok=True)
    
    # Save summary to file
    with open('outputs/threshold_analysis/threshold_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("THRESHOLD EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Threshold':<12} {'Accuracy':<12} {'Recall(Mal)':<15} {'Spec(Ben)':<15} {'F1':<10}\n")
        f.write("-"*80 + "\n")
        
        for r in results:
            f.write(f"{r['threshold']:<12.2f} {r['accuracy']:<12.4f} {r['recall_malignant']:<15.4f} "
                   f"{r['specificity_benign']:<15.4f} {r['f1_score']:<10.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("BEST THRESHOLDS BY METRIC\n")
        f.write("="*80 + "\n")
        f.write(f"Best Accuracy:     Threshold = {best_accuracy['threshold']:.2f}, "
               f"Accuracy = {best_accuracy['accuracy']:.4f}\n")
        f.write(f"Best Recall (Mal): Threshold = {best_recall['threshold']:.2f}, "
               f"Recall = {best_recall['recall_malignant']:.4f}\n")
        f.write(f"Best F1 Score:     Threshold = {best_f1['threshold']:.2f}, "
               f"F1 = {best_f1['f1_score']:.4f}\n")
        f.write(f"Best Balanced:     Threshold = {best_balanced['threshold']:.2f}, "
               f"Recall+Spec = {best_balanced['recall_malignant'] + best_balanced['specificity_benign']:.4f}\n")
    
    print(f"\n[OK] Summary saved to: outputs/threshold_analysis/threshold_summary.txt")
    
    # Plot metrics vs threshold
    plot_metrics_vs_threshold(results)
    
    print("\n" + "="*80)
    print("[OK] EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: outputs/threshold_analysis/")
    print(f"  - threshold_summary.txt")
    print(f"  - metrics_vs_threshold.png")
    print(f"  - confusion_matrix_thX.XX.png (for each threshold)")


if __name__ == "__main__":
    main()
