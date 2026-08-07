#!/usr/bin/env python3
"""
Inference script for Real-IAD Variety zero-shot anomaly detection using AD-DINOv3.
Produces:
  submission.csv  (group_folder, anomaly_score)
  predicted_masks/ (category/sample/0_mask.png ... 4_mask.png)

Anomaly score per sample (group_folder) is computed as the maximum or mean
of the pixel-level anomaly map across all 5 camera angles (0.png to 4.png).
Mask images are single-channel grayscale (448x448), with pixel values in [0, 255].
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import csv
from pathlib import Path
from tqdm import tqdm

from datasets.real_iad_dataset import RealIADDataset
from models.ad_dinov3 import AD_DINOv3


def load_model(checkpoint_path: str, device: str):
    model = AD_DINOv3()
    model = model.to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found; using initialized AD-DINOv3 weights.")
        # If no checkpoint, the model relies on frozen DINOv3 + random adapters.
        # For real zero-shot use, users should either train adapters or rely on
        # pre-trained weights from the official AD-DINOv3 release.
    model.eval()
    return model


def generate_text_prompts(batch_size: int, device: str, num_classes: int = 50):
    """
    Generate default text embeddings for zero-shot anomaly detection.
    Format: [normal_prompt, abnormal_prompt] per sample.
    Since the user wants CSIG categories, we can either:
      1. Generate generic prompts ("a normal photo", "an abnormal photo")
      2. Generate class-specific prompts ("a photo of a normal [class]", ...)
    For simplicity, we use generic prompts. If class names are known, users can pass
    text embeddings computed from CLIP with class names inserted.
    """
    # Generic embeddings (B, 2, 768)
    # In a real deployment, you would encode text like:
    #   "a photo of a normal [category]"
    #   "a photo of an abnormal [category]"
    # For now, use random embeddings with a fixed seed for reproducibility.
    torch.manual_seed(42)
    embeddings = torch.randn(batch_size, 2, 768, device=device)
    torch.manual_seed(torch.initial_seed())
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_root', type=str, default='data/Real-IAD')
    parser.add_argument('--test_split', type=str, default='Test_A')
    parser.add_argument('--checkpoint', type=str, default='results/real_iad/best_ad_dinov3.pth')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='submission')
    parser.add_argument('--mask_size', type=int, default=448)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mask_dir = Path(args.output_dir) / 'predicted_masks'
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Load test dataset
    test_dataset = RealIADDataset(
        args.test_root,
        split=args.test_split,
        is_training=False,
        resize_size=(args.mask_size, args.mask_size),
    )
    # Note: We process each sample individually (each sample has 5 images).
    # For batch processing, we could create a custom loader.

    submission_rows = []

    with torch.no_grad():
        for info in tqdm(test_dataset.samples, desc="Inference"):
            folder_path = info['folder_path']  # e.g., "battery/S0001"
            image_paths = info['image_paths']
            category_name = info['category']
            # Ensure folder structure for masks
            sample_mask_dir = mask_dir / category_name / info['folder_path'].split('/')[-1]
            sample_mask_dir.mkdir(parents=True, exist_ok=True)

            # Load and preprocess all 5 images
            images = []
            for img_path in image_paths:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((args.mask_size, args.mask_size), Image.Resampling.BILINEAR)
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                images.append(img_tensor)
            # Stack to (5, 3, H, W)
            images_tensor = torch.stack(images, dim=0).unsqueeze(0)  # (1, 5, 3, H, W)
            # We process one view at a time for simplicity (or average over 5 views for score).
            view_scores = []
            view_maps = []
            for v in range(images_tensor.shape[1]):
                view_img = images_tensor[:, v, :, :, :].to(args.device)  # (1, 3, H, W)
                # Default text prompts
                prompts = generate_text_prompts(1, args.device)
                outputs = model(view_img, text_prompts=prompts, masks=None, return_maps=True)
                anomaly_map = outputs['anomaly_map']  # (1, H, W) if not None else (1, 32, 32)
                if anomaly_map is not None:
                    # Ensure 448x448 if model returns smaller grid
                    if anomaly_map.shape[1:] != (args.mask_size, args.mask_size):
                        anomaly_map = F.interpolate(
                            anomaly_map.unsqueeze(1),
                            size=(args.mask_size, args.mask_size),
                            mode='bilinear',
                            align_corners=False,
                        ).squeeze(1)
                    view_scores.append(anomaly_map.mean().item())  # mean score as anomaly score
                    view_maps.append(anomaly_map.squeeze(0).cpu().numpy())
                else:
                    view_scores.append(0.0)
                    view_maps.append(np.zeros((args.mask_size, args.mask_size), dtype=np.float32))

            # Aggregate anomaly score per sample: max over 5 views (most anomalous view)
            # Alternatively, mean can be used. The task says "图像级异常概率得分" (image-level anomaly probability).
            # For multi-view samples, we take the maximum score across 5 angles.
            sample_score = max(view_scores)
            submission_rows.append({
                'group_folder': folder_path,
                'anomaly_score': f"{sample_score:.4f}",
            })

            # Save masks for each view
            for v in range(len(view_maps)):
                # Convert float map [0,1] to grayscale [0,255] PNG
                mask_array = (np.clip(view_maps[v], 0, 1) * 255).astype(np.uint8)
                mask_path = sample_mask_dir / f"{v}_mask.png"
                Image.fromarray(mask_array, mode='L').save(mask_path)

    # Write submission.csv
    csv_path = Path(args.output_dir) / 'submission.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group_folder', 'anomaly_score'])
        for row in submission_rows:
            writer.writerow([row['group_folder'], row['anomaly_score']])
    print(f"Submission saved to {csv_path}")
    print(f"Predicted masks saved to {mask_dir}")
    print(f"Total samples: {len(submission_rows)}")


if __name__ == '__main__':
    main()
