#!/usr/bin/env python3
"""
Compute the final submission score S based on user-defined formula:
  S = 100 * (0.4 * S_cls + 0.6 * S_seg)
Where:
  S_cls = macro-average of image-level I-AUROC and I-AUPR over categories
  S_seg = macro-average of pixel-level P-AUROC, P-AUPR, P-F1max over categories

This script expects prediction results and optionally ground-truth labels/masks
for evaluation. Without real ground truth, it can only compute approximate scores
if predictions are available.
"""
import numpy as np
from pathlib import Path
import csv
from collections import defaultdict


def compute_score(s_cls: float, s_seg: float) -> float:
    return 100.0 * (0.4 * s_cls + 0.6 * s_seg)


def load_submission(submission_path: str):
    rows = []
    with open(submission_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'group_folder': row['group_folder'],
                'anomaly_score': float(row['anomaly_score'])
            })
    return rows


def main():
    # Example usage (requires ground truth to compute real metrics)
    submission_path = 'submission/submission.csv'
    if Path(submission_path).exists():
        rows = load_submission(submission_path)
        print(f"Loaded {len(rows)} predictions from {submission_path}")
        # Compute approximate average score for demonstration
        avg_score = np.mean([r['anomaly_score'] for r in rows])
        print(f"Average predicted anomaly score: {avg_score:.4f}")
    else:
        print("No submission.csv found. Run infer_real_iad.py first.")


if __name__ == '__main__':
    main()
