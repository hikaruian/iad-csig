#!/usr/bin/env python3
"""
Training script for CSIG 50-class multi-class image classification.
Uses AD-DINOv3 as feature extractor (frozen DINOv3 + light adapter + projection)
and trains a simple linear classifier head on top.
Also supports fine-tuning the adapter and projection head.
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

from datasets.csig_dataset import CSGIDataset
from models.ad_dinov3 import AD_DINOv3
from utils.metrics import classification_accuracy


class CSIGClassifier(nn.Module):
    def __init__(self, feature_dim: int = 512, num_classes: int = 50):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def train_epoch(model, classifier, loader, optimizer, criterion, device):
    model.train()
    classifier.train()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch[0].to(device)  # (B, 3, 448, 448)
        labels = batch[1].to(device)
        optimizer.zero_grad()
        # Extract features from AD-DINOv3
        features = model.forward_classification(images)  # (B, align_dim=512)
        logits = classifier(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_acc += classification_accuracy(logits.detach().cpu().numpy(), labels.cpu().numpy()) * images.size(0)
        count += images.size(0)
    return total_loss / count, total_acc / count


def validate_epoch(model, classifier, loader, criterion, device):
    model.eval()
    classifier.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            images = batch[0].to(device)
            labels = batch[1].to(device)
            features = model.forward_classification(images)
            logits = classifier(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_acc += classification_accuracy(logits.cpu().numpy(), labels.cpu().numpy()) * images.size(0)
            count += images.size(0)
    return total_loss / count, total_acc / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root', type=str, default='data/CSIG/Train')
    parser.add_argument('--val_root', type=str, default='data/CSIG/Test_A')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='results/csig')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize AD-DINOv3 (frozen backbone, trainable adapter and projection)
    model = AD_DINOv3()
    model = model.to(args.device)
    # Note: By default our AD_DINOv3 keeps adapters and projections trainable
    # (only DINOv3 backbone and CLIP encoder are frozen).

    classifier = CSIGClassifier(args.feature_dim, args.num_classes).to(args.device)

    # Data loaders
    train_dataset = CSGIDataset(args.train_root, resize_size=(448, 448), transform=None)
    val_dataset = CSGIDataset(args.val_root, resize_size=(448, 448), transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, classifier, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc = validate_epoch(model, classifier, val_loader, criterion, args.device)
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model (val_acc={val_acc:.4f})")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
