"""CSIG / Real-IAD Variety multi-view dataset.

Expected layout
---------------
<root>/<class_name>/Sxxxx/{0,1,2,3,4}.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image

try:
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:  # helpers (discover_samples / group_folder) work without torch
    Dataset = object  # type: ignore
    transforms = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIEWS = (0, 1, 2, 3, 4)

SEEN_CATEGORIES = {
    "3_adapter",
    "DVD_switch",
    "D_sub_connector",
    "PLCC_socket",
    "VR_joystick",
    "accurate_detection_switch",
    "battery",
    "blade_switch",
    "boost_converter_module",
    "button_battery_holder",
    "circuit_breaker",
    "connector_housing_female",
    "crimp_st_cable_mount_box",
    "dc_jack",
    "dc_power_connector",
    "detection_switch",
    "effect_transistor",
    "electronic_watch_movement",
    "ffc_connector_plug",
    "ingot_buckle",
    "laser_diode",
    "lego_pin_connector_plate",
    "limit_switch",
    "lithium_battery_plug",
    "littel_fuse",
    "lock",
    "miniature_lifting_motor",
    "mobile_charging_connector",
    "motor_bracket",
    "motor_gear_reducer",
    "motor_plug",
    "pencil_sharpener",
    "pinboard_connector",
    "potentiometer",
    "power_jack",
    "power_strip_socket",
    "purple_clay_pot",
    "retaining_ring",
    "rheostat",
    "self_lock_switch",
    "silicon_cell_sensor",
    "single_switch",
    "smd_receiver_module",
    "suction_cup",
    "toy_tire",
    "travel_switch",
    "vacuum_switch",
    "vehicle_harness_conductor",
    "vibration_motor",
    "wireless_receiver_module",
}


def build_transform(image_size: int = 448, is_train: bool = False):
    if transforms is None:
        raise ImportError("torchvision is required to build image transforms")
    ops: List = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    # Official INP-Former uses resize + normalize only. No geometric
    # augmentation: masks must stay aligned with the 448×448 canvas.
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def discover_samples(root: Path) -> List[Tuple[str, str, Path]]:
    """Return (category, sample_id, sample_dir) sorted."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    samples = []
    for cat_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sample_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            pngs = list(sample_dir.glob("*.png"))
            if not pngs:
                continue
            samples.append((cat_dir.name, sample_dir.name, sample_dir))
    if not samples:
        raise RuntimeError(f"No samples found under {root}")
    return samples


def group_folder(category: str, sample_id: str) -> str:
    return f"{category}/{sample_id}"


def view_path(sample_dir: Path, view_id: int) -> Path:
    for name in (
        f"{view_id}.png",
        f"{view_id}.PNG",
        f"{view_id:02d}.png",
        f"{view_id}.jpg",
        f"{view_id}.jpeg",
    ):
        p = sample_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"Missing view {view_id} in {sample_dir}")


class CSIGImageDataset(Dataset):
    """Flattens every view into an independent training image (unsupervised)."""

    def __init__(self, root: str, transform: Optional[Callable] = None, image_size: int = 448):
        self.root = Path(root)
        self.transform = transform or build_transform(image_size, is_train=True)
        self.items: List[Tuple[str, str, int, Path]] = []
        for cat, sid, sdir in discover_samples(self.root):
            for v in VIEWS:
                try:
                    self.items.append((cat, sid, v, view_path(sdir, v)))
                except FileNotFoundError:
                    continue
        if not self.items:
            raise RuntimeError(f"No view images found under {root}")
        self.classes = sorted({c for c, _, _, _ in self.items})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cat, sid, view_id, path = self.items[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")
            img.load()
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[cat]


class CSIGSampleDataset(Dataset):
    """One item = one physical sample with 5 views (used at inference)."""

    def __init__(self, root: str, transform: Optional[Callable] = None, image_size: int = 448):
        self.root = Path(root)
        self.transform = transform or build_transform(image_size, is_train=False)
        self.samples = discover_samples(self.root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cat, sid, sdir = self.samples[idx]
        views = []
        for v in VIEWS:
            with Image.open(view_path(sdir, v)) as im:
                img = im.convert("RGB")
                img.load()
            views.append(self.transform(img))
        # stack -> (5, 3, H, W)
        import torch

        return {
            "images": torch.stack(views, dim=0),
            "group_folder": group_folder(cat, sid),
            "category": cat,
            "sample_id": sid,
        }


def list_group_folders(root: str) -> List[str]:
    return [group_folder(c, s) for c, s, _ in discover_samples(Path(root))]
