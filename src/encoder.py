"""Frozen DINOv2-with-registers encoder.

Official INP-Former / Dinomaly backbone. Intermediate tokens are taken from
the same middle-layer groups as the papers (ViT-B: blocks 2-9).

Under DDP, rank 0 downloads weights first so two processes do not corrupt
the torch.hub / timm cache.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn

from .dist_utils import barrier, env_world, is_main


ENCODER_PRESETS = {
    "dinov2reg_vit_small_14": {
        "hub": "dinov2_vits14_reg",
        "timm": "vit_small_patch14_reg4_dinov2.lvd142m",
        "embed_dim": 384,
        "num_heads": 6,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
        "patch_size": 14,
    },
    "dinov2reg_vit_base_14": {
        "hub": "dinov2_vitb14_reg",
        "timm": "vit_base_patch14_reg4_dinov2.lvd142m",
        "embed_dim": 768,
        "num_heads": 12,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
        "patch_size": 14,
    },
    "dinov2reg_vit_large_14": {
        "hub": "dinov2_vitl14_reg",
        "timm": "vit_large_patch14_reg4_dinov2.lvd142m",
        "embed_dim": 1024,
        "num_heads": 16,
        "target_layers": [4, 6, 8, 10, 12, 14, 16, 18],
        "patch_size": 14,
    },
}


def prefetch_encoder_weights(name: str, source: str = "auto", stamp_dir: str = "runs/_prefetch") -> None:
    """Download DINOv2 weights BEFORE init_process_group.

    Rank 0 writes a stamp file; other ranks poll it. Avoids NCCL timeout on first run.
    Safe to call when WORLD_SIZE=1 (no-op besides a single load into cache).
    """
    rank, _, world_size = env_world()
    if world_size <= 1:
        return
    stamp = Path(stamp_dir) / f"{name}.{source}.ready"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    fail = stamp.with_suffix(".fail")
    if rank == 0:
        try:
            _ = DinoV2Encoder(name, source=source)
            del _
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stamp.write_text("ok", encoding="utf-8")
        except Exception as exc:
            fail.write_text(repr(exc), encoding="utf-8")
            raise
        return
    deadline = time.time() + 3600
    while time.time() < deadline:
        if stamp.exists():
            return
        if fail.exists():
            raise RuntimeError(f"Rank 0 failed to prefetch {name}: {fail.read_text(encoding='utf-8')}")
        time.sleep(1.0)
    raise TimeoutError(
        f"Rank {rank} waited 1h for rank 0 to prefetch {name}. "
        "Check network access to dl.fbaipublicfiles.com / torch.hub."
    )


class DinoV2Encoder(nn.Module):
    """Wraps a pretrained DINOv2 model and exposes intermediate patch tokens."""

    def __init__(self, name: str = "dinov2reg_vit_base_14", source: str = "auto"):
        super().__init__()
        if name not in ENCODER_PRESETS:
            raise ValueError(f"Unknown encoder {name}. Choose from {list(ENCODER_PRESETS)}")
        self.cfg = ENCODER_PRESETS[name]
        self.name = name
        self.backend = None
        self.model = self._load_synced(source)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def embed_dim(self) -> int:
        return self.cfg["embed_dim"]

    @property
    def num_heads(self) -> int:
        return self.cfg["num_heads"]

    @property
    def target_layers(self) -> List[int]:
        return list(self.cfg["target_layers"])

    @property
    def patch_size(self) -> int:
        return int(self.cfg["patch_size"])

    def _load_synced(self, source: str):
        # Rank 0 populates the cache; others wait, then load from disk.
        # try/finally so a failed download cannot leave the other rank stuck on barrier.
        if is_main():
            try:
                model = self._load(source)
            finally:
                barrier()
            return model
        barrier()
        return self._load(source)

    def _load(self, source: str):
        errors = []
        order = ["hub", "timm"] if source == "auto" else [source]
        for kind in order:
            try:
                if kind == "hub":
                    model = torch.hub.load(
                        "facebookresearch/dinov2",
                        self.cfg["hub"],
                        pretrained=True,
                        trust_repo=True,
                    )
                    self.backend = "hub"
                    return model
                if kind == "timm":
                    import timm

                    model = timm.create_model(
                        self.cfg["timm"],
                        pretrained=True,
                        dynamic_img_size=True,
                        num_classes=0,
                    )
                    self.backend = "timm"
                    return model
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{kind}: {exc}")
        raise RuntimeError(
            "Failed to load DINOv2 weights. Install torch + internet, or timm.\n"
            + "\n".join(errors)
        )

    def train(self, mode: bool = True):
        # Encoder must stay in eval (frozen LN / no dropout) even if parent is train().
        super().train(False)
        self.model.eval()
        return self

    def forward_features(self, x: torch.Tensor, layers: Sequence[int] | None = None) -> List[torch.Tensor]:
        """Return a list of patch-token tensors, each (B, N, C)."""
        layers = list(layers) if layers is not None else self.target_layers
        # no_grad (not inference_mode): features must be cloneable into the train graph.
        with torch.no_grad():
            feats = self._forward_features(x, layers)
        return [f.detach().clone() for f in feats]

    def _forward_features(self, x: torch.Tensor, layers: Sequence[int]) -> List[torch.Tensor]:
        if self.backend == "hub":
            feats = self.model.get_intermediate_layers(
                x, n=list(layers), reshape=False, return_class_token=False, norm=False
            )
            return [f.contiguous() for f in feats]

        intermediates = self.model.forward_intermediates(
            x,
            indices=list(layers),
            norm=False,
            stop_early=True,
            output_fmt="NLC",
            intermediates_only=True,
        )
        cleaned = []
        prefix = 1 + int(getattr(self.model, "num_reg_tokens", getattr(self.model, "num_register_tokens", 0)))
        n_expected = (x.shape[-2] // self.patch_size) * (x.shape[-1] // self.patch_size)
        for feat in intermediates:
            if feat.dim() == 4:
                feat = feat.flatten(2).transpose(1, 2)
            elif feat.shape[1] != n_expected:
                feat = feat[:, prefix:, :]
            cleaned.append(feat.contiguous())
        return cleaned
