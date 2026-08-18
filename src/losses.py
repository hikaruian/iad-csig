"""Soft Mining Loss + INP Coherence Loss (INP-Former).

AMP-safe: factor is computed in fp32 and clamped so T4 fp16 grads do not overflow.
Hooks are only attached when the decoder tensor actually requires grad.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F


def _scale_grad(x, factor):
    # factor: (B, 1, H, W) fp32; x: grad of decoder feature
    if factor.dtype != x.dtype:
        factor = factor.to(dtype=x.dtype)
    return x * factor.expand_as(x)


def global_cosine_hm_adaptive(a, b, y: float = 3.0) -> torch.Tensor:
    """Soft mining: up-weight hard-to-reconstruct normal tokens.

    Matches INP-Former utils.global_cosine_hm_adaptive, with numerical guards.
    """
    if len(a) == 0:
        raise ValueError("empty feature list")
    loss = a[0].new_zeros(())
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        # point-wise cosine on channel dim, always in fp32 for a stable threshold
        with torch.no_grad():
            point_dist = 1.0 - F.cosine_similarity(a_.float(), b_.detach().float(), dim=1)
            point_dist = point_dist.unsqueeze(1)
            mean_dist = point_dist.mean().clamp_min(1e-6)
            factor = (point_dist / mean_dist).pow(y).clamp_(0.0, 32.0)

        # official: one global cosine per image, then mean over batch
        rec = 1.0 - F.cosine_similarity(
            a_.reshape(a_.shape[0], -1).float(),
            b_.reshape(b_.shape[0], -1).float(),
            dim=1,
        )
        loss = loss + rec.mean()
        if b_.requires_grad:
            b_.register_hook(partial(_scale_grad, factor=factor))
    return loss / len(a)


def total_loss(en, de, gather_loss, gather_weight: float = 0.2, y: float = 3.0) -> torch.Tensor:
    rec = global_cosine_hm_adaptive(en, de, y=y)
    return rec + gather_weight * gather_loss
