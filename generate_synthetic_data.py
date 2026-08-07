#!/usr/bin/env python3
"""
Synthetic dataset generator for demonstration purposes.
Creates synthetic CSIG-style and Real-IAD-style data directories
with random images and dummy masks so that the pipeline can be tested end-to-end.
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import random

def generate_synthetic_csig(output_root="data/CSIG", num_classes=10, samples_per_class=5, images_per_sample=5):
    os.makedirs(output_root + "/Train", exist_ok=True)
    for cls_idx in range(num_classes):
        cls_name = f"class_{cls_idx:02d}"
        cls_dir = os.path.join(output_root, "Train", cls_name)
        for s in range(samples_per_class):
            s_dir = os.path.join(cls_dir, f"S{(s+1):04d}")
            os.makedirs(s_dir, exist_ok=True)
            for i in range(images_per_sample):
                # Random colored square image
                img = Image.new('RGB', (256, 256), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                draw = ImageDraw.Draw(img)
                x0, y0 = random.randint(20, 120), random.randint(20, 120)
                x1, y1 = x0 + random.randint(40, 80), y0 + random.randint(40, 80)
                draw.ellipse([x0, y0, x1, y1], fill=(255, 255, 0))
                img.save(os.path.join(s_dir, f"{i}.png"))
    print(f"Synthetic CSIG Train data created at {output_root}/Train")


def generate_synthetic_real_iad(output_root="data/Real-IAD", num_classes=10, samples_per_class=5):
    for split in ["Train", "Test_A"]:
        split_dir = os.path.join(output_root, split)
        os.makedirs(split_dir, exist_ok=True)
        for cls_idx in range(num_classes):
            cls_name = f"component_{cls_idx:03d}"
            cls_dir = os.path.join(split_dir, cls_name)
            for s in range(samples_per_class):
                s_dir = os.path.join(cls_dir, f"S{(s+1):04d}")
                image_dir = os.path.join(s_dir)
                mask_dir = os.path.join(s_dir, "masks")
                os.makedirs(image_dir, exist_ok=True)
                os.makedirs(mask_dir, exist_ok=True)
                for i in range(5):
                    img = Image.new('RGB', (448, 448), color=(random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)))
                    # Add a random anomaly region for demonstration
                    draw = ImageDraw.Draw(img)
                    x0, y0 = random.randint(50, 300), random.randint(50, 300)
                    x1, y1 = x0 + random.randint(20, 100), y0 + random.randint(20, 100)
                    draw.ellipse([x0, y0, x1, y1], fill=(255, 0, 0))
                    img.save(os.path.join(image_dir, f"{i}.png"))
                # Dummy mask (grayscale)
                mask = Image.new('L', (448, 448), 0)
                draw_m = ImageDraw.Draw(mask)
                x0, y0 = random.randint(50, 300), random.randint(50, 300)
                x1, y1 = random.randint(x0+20, 420), random.randint(y0+20, 420)
                draw_m.ellipse([x0, y0, x1, y1], fill=255)
                mask.save(os.path.join(mask_dir, f"{i}_mask.png"))
    print(f"Synthetic Real-IAD data created at {output_root}")


if __name__ == '__main__':
    generate_synthetic_csig()
    generate_synthetic_real_iad()
