#!/usr/bin/env python3
"""
Training script for Real-IAD Variety zero-shot anomaly detection using AD-DINOv3.
Uses an auxiliary dataset (e.g., MVTec AD style or synthetic) for training adapters,
but the framework supports zero-shot evaluation on unseen categories.
In this task, we assume the user provides training data with some categories
and evaluates on Test_A (with both seen and potentially unseen categories).

Training objective (as per paper):
  L = lambda_CM * L_CM + lambda_AACM * L_AACM
where L_CM is focal + dice on cross-modal anomaly map,
and L_AACM is focal + dice on CLS-patch similarity guided by mask.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from datasets.real_iad_dataset import RealIADDataset
from models.ad_dinov3 import AD_DINOv3


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_cm = 0.0
    total_aacm = 0.0
    total_loss = 0.0
    count = 0
    for batch in tqdm(loader, desc="Train Real-IAD", leave=False):
        images = batch['images'].to(device)  # (B, 5, 3, 448, 448)
        # For simplicity, use the first view (0.png) for training.
        # In practice, you could aggregate over all 5 views or train on each separately.
        view_image = images[:, 0, :, :, :]  # (B, 3, 448, 448)
        # For training with masks: we need masks. In the dataset loader above,
        # we didn't include masks. For real use, masks should be loaded from
        # a corresponding mask directory or generated from annotations.
        # Here we assume masks are not available during standard dataset loading
        # (as the dataset only contains images for CSIG/Real-IAD without annotations),
        # but for demonstration we create dummy masks.
        masks = torch.zeros(view_image.size(0), view_image.size(2), view_image.size(3), device=device)
        optimizer.zero_grad()
        # Default text prompts for anomaly detection
        # Format: [normal_prompt, abnormal_prompt]
        # We create dummy tokenized text as embeddings (B, 2, D) using random initialization
        # for demonstration. In real use, users should load CLIP-encoded prompts.
        B = view_image.size(0)
        dummy_text = torch.randn(B, 2, 768, device=device)
        outputs = model(view_image, text_prompts=dummy_text, masks=masks, return_maps=True)
        loss = outputs['total_loss']
        # If masks are all zeros, the loss will train to predict no anomalies,
        # which is not meaningful without real annotations. For real training,
        # replace `masks` with actual ground-truth binary masks.
        loss.backward()
        optimizer.step()
        total_cm += outputs['cm_loss'].item() * B
        total_aacm += outputs['aacm_loss'].item() * B
        total_loss += loss.item() * B
        count += B
    return {
        'loss': total_loss / count,
        'cm_loss': total_cm / count,
        'aacm_loss': total_aacm / count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', type=str, default='data/Real-IAD')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='results/real_iad')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize AD-DINOv3
    model = AD_DINOv3()
    model = model.to(args.device)

    # Load dataset (Train split only for training adapter)
    train_dataset = RealIADDataset(args.train_root, split='Train', is_training=True, resize_size=(448, 448))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    optimizer = optim.AdamW(
        # Only train adapters, projection heads, and any trainable parameters in AD_DINOv3
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        metrics = train_epoch(model, train_loader, optimizer, args.device)
        scheduler.step()
        print(f"Loss: {metrics['loss']:.4f} | CM: {metrics['cm_loss']:.4f} | AACM: {metrics['aacm_loss']:.4f}")
        if metrics['loss'] < best_loss:
            best_loss = metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(args.save_dir, 'best_ad_dinov3.pth'))
            print(f"Saved best model (loss={best_loss:.4f})")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
