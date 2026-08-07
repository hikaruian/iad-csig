"""
Anomaly-Aware Calibration Module (AACM) from AD-DINOv3.
Guides the CLS token to attend to anomalous regions using mask supervision.
Loss = focal_loss + dice_loss on similarity between adapted CLS and patch tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AACM(nn.Module):
    def __init__(self):
        super().__init__()
        # AACM is primarily a training-time regularizer; no learnable params needed
        # beyond the adapter that produces adapted features.
        pass

    def similarity_distribution(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between CLS token and each patch token.
        Args:
            cls_token: (B, 1, D) or (B, D)
            patch_tokens: (B, N, D) where N = num_patches
        Returns:
            probs: (B, N) after softmax over patches (attention distribution over patches)
        """
        # Normalize
        cls_token = F.normalize(cls_token, p=2, dim=-1)
        patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)
        # Similarity: (B, 1, D) @ (B, D, N) -> (B, 1, N) -> squeeze
        if cls_token.dim() == 3:
            cls_token = cls_token.squeeze(1)  # (B, D)
        sim = torch.bmm(cls_token.unsqueeze(1), patch_tokens.transpose(1, 2)).squeeze(1)  # (B, N)
        probs = F.softmax(sim, dim=-1)
        return probs

    @staticmethod
    def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0):
        """
        Focal loss for binary mask (patch-level).
        pred: (B, N) in [0,1] (probability of being anomalous region according to CLS attention)
        target: (B, H*W) or (B, N) binary mask downsampled to patch grid.
        """
        # Treat target as binary mask for patches
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = target * pred + (1 - target) * (1 - pred)
        focal = alpha * (1 - pt).pow(gamma) * bce
        return focal.mean()

    @staticmethod
    def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
        """
        Dice loss for patch-level mask.
        """
        pred_flat = pred.reshape(pred.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = 1 - ((2 * intersection + smooth) / (union + smooth)).mean()
        return dice

    def loss(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute AACM loss = focal + dice between CLS-patch similarity and ground-truth mask.
        Args:
            cls_token: (B, D) adapted CLS token.
            patch_tokens: (B, N, D) adapted patch tokens.
            mask: (B, H, W) binary mask at original resolution; will be downsampled to patch grid.
        Returns:
            loss scalar.
        """
        # Compute similarity distribution over patches
        sim = self.similarity_distribution(cls_token, patch_tokens)  # (B, N)
        # Downsample mask to patch grid (approximate by average pooling concept)
        # For simplicity in general case, assume mask can be reshaped or resized to (B, sqrt(N), sqrt(N))
        # If mask is provided at image level, we interpolate to patch grid.
        B, N = sim.shape
        # Assume square grid: H_patches = W_patches = int(sqrt(N))
        grid_size = int(N ** 0.5)
        mask_down = F.interpolate(mask.unsqueeze(1).float(), size=(grid_size, grid_size), mode='bilinear', align_corners=False).squeeze(1)
        mask_down = mask_down.reshape(B, -1)  # (B, N)
        loss_focal = self.focal_loss(sim, mask_down)
        loss_dice = self.dice_loss(sim, mask_down)
        return loss_focal + loss_dice
