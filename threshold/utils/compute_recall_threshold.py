from sklearn.metrics import recall_score
import torch

def compute_recall_threshold(model, dataloader, device, threshold=0.5):
    """
    Compute recall for threshold-based curriculum learning.
    Handles both old format (imgs, labels) and new format (imgs, labels, is_syn, jsd_scores).
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            # Handle both old format (imgs, labels) and new format (imgs, labels, is_syn, jsd_scores)
            if len(batch) == 4:
                imgs, labels, _, _ = batch
            else:
                imgs, labels = batch
            
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            preds = (probs > threshold).long()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return recall_score(y_true, y_pred, pos_label=1)
