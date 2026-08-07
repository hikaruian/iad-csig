"""
Real-IAD Variety dataset loader (deep cleaned subset as per task description).
Directory structure:
  Real-IAD/
    Train/  (only 50 categories, only normal samples)
      <category_name>/
        SXXXX/
          0.png .. 4.png
    Test_A/  (50 categories, normal + abnormal samples)
      <category_name>/
        SXXXX/
          0.png .. 4.png

Each sample folder (SXXXX) represents one physical sample with 5 camera angles.
Images are RGB PNG with variable sizes; resize to 448x448 as required for masks.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np


class RealIADDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "Train",  # "Train" or "Test_A"
        transform=None,
        resize_size: tuple = (448, 448),
        is_training: bool = True,
    ):
        self.root = Path(root) / split
        self.transform = transform
        self.resize_size = resize_size
        self.is_training = is_training
        # Discover categories
        self.categories = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.category_to_idx = {c: i for i, c in enumerate(self.categories)}
        self.samples = []  # list of (sample_path_str, category, image_paths)
        # Build sample list
        for cat in self.categories:
            cat_path = self.root / cat
            for sample_dir in sorted(cat_path.iterdir()):
                if not sample_dir.is_dir():
                    continue
                if not sample_dir.name.startswith('S'):
                    continue
                # Find all 5 images
                img_paths = sorted(sample_dir.glob('*.png'))
                # Filter only 0.png .. 4.png if present, else take all png
                img_paths = [p for p in img_paths if p.name.startswith(('0.', '1.', '2.', '3.', '4.')) or p.suffix == '.png']
                # Ensure 5 images; if some missing, take whatever exists
                if len(img_paths) > 0:
                    self.samples.append({
                        'folder_path': f"{cat}/{sample_dir.name}",  # relative path string
                        'category': cat,
                        'image_paths': img_paths,
                        'label': 0 if is_training else None,  # Training: assume all normal (ZSAD setting)
                    })
        print(f"[RealIADDataset] Split={split}, Categories={len(self.categories)}, Samples={len(self.samples)}, Training={is_training}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        images = []
        for img_path in info['image_paths']:
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.resize_size, Image.Resampling.BILINEAR)
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = (image - mean) / std
            images.append(image)
        # Stack to (5, 3, H, W) or return list
        images_tensor = torch.stack(images, dim=0)  # (num_angles, 3, H, W)
        return {
            'images': images_tensor,
            'folder_path': info['folder_path'],
            'category': info['category'],
            'label': info['label'],
            'num_angles': images_tensor.shape[0],
        }
