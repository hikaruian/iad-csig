"""
Evaluation metrics for industrial anomaly detection (image-level and pixel-level).
Used for both CSIG multi-class classification and Real-IAD anomaly detection.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def image_auroc(scores, labels):
    """
    Image-level AUROC (binary: 0=normal, 1=anomalous).
    Args:
        scores: array of anomaly scores (higher = more anomalous)
        labels: binary labels (0/1)
    Returns:
        float AUROC
    """
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.5


def image_aupr(scores, labels):
    """Image-level AUPR (anomalous class = positive)."""
    try:
        return average_precision_score(labels, scores)
    except ValueError:
        return 0.0


def pixel_auroc(anomaly_maps, ground_truth_masks):
    """
    Pixel-level AUROC.
    Args:
        anomaly_maps: np.array (B, H, W) with anomaly scores in [0,1]
        ground_truth_masks: np.array (B, H, W) binary masks
    Returns:
        float
    """
    maps_flat = anomaly_maps.reshape(-1)
    masks_flat = ground_truth_masks.reshape(-1)
    try:
        return roc_auc_score(masks_flat, maps_flat)
    except ValueError:
        return 0.5


def pixel_aupr(anomaly_maps, ground_truth_masks):
    """Pixel-level AUPR (anomalous pixel = positive)."""
    maps_flat = anomaly_maps.reshape(-1)
    masks_flat = ground_truth_masks.reshape(-1)
    try:
        return average_precision_score(masks_flat, maps_flat)
    except ValueError:
        return 0.0


def pixel_f1_max(anomaly_maps, ground_truth_masks):
    """
    Pixel-level maximum F1 score across thresholds.
    """
    best_f1 = 0.0
    maps_flat = anomaly_maps.reshape(-1)
    masks_flat = ground_truth_masks.reshape(-1)
    # Try a set of thresholds
    thresholds = np.linspace(0, 1, 50)
    for t in thresholds:
        pred = (maps_flat >= t).astype(int)
        try:
            f1 = f1_score(masks_flat, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
        except Exception:
            continue
    return float(best_f1)


def classification_accuracy(logits, labels):
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds == labels))
