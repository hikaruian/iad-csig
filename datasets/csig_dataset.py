"""
CSIG Dataset loader for 50-class multi-class image classification.
Directory structure (as given in prompt):
  CSIG/
    Train/
      <class_name>/
        SXXXX/
          0.png .. 4.png
    Test_A/
      (same structure)
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np


class CSGIDataset(Dataset):
    def __init__(self, root: str, transform=None, resize_size: tuple = (448, 448)):
        self.root = Path(root)
        self.transform = transform
        self.resize_size = resize_size
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_path = self.root / cls
            if not cls_path.is_dir():
                continue
            for sample_dir in sorted(cls_path.iterdir()):
                if not sample_dir.is_dir():
                    continue
                # Filter directories starting with 'S'
                if not sample_dir.name.startswith('S'):
                    continue
                for img_path in sorted(sample_dir.glob('*.png')):
                    self.samples.append((img_path, self.class_to_idx[cls]))
        print(f"[CSGIDataset] Loaded {len(self.samples)} images from {len(self.classes)} classes at {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        # Resize to handle variable image sizes (400x400 to 4400x4400)
        image = image.resize(self.resize_size, Image.Resampling.BILINEAR)
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert PIL to tensor and normalize
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            # Standard ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
        return image, label, str(img_path)
