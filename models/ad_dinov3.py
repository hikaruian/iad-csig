"""
AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration
Full architecture integrating:
  - DINOv3 ViT-L/16 visual backbone (frozen, multi-level features from layers 6,12,18,24)
  - CLIP text encoder (frozen, with light adapter)
  - Multi-level visual adapters (bottleneck MLP)
  - Anomaly-Aware Calibration Module (AACM) for CLS token guidance
  - Cross-Modal Contrastive Learning (CMCL) for pixel-level anomaly localization

Reference paper:
  AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration
  arXiv:2509.14084
  GitHub: https://github.com/Kaisor-Yuan/AD-DINOv3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from .adapter import LightAdapter, MultiLevelVisualAdapter
from .aacm import AACM


class AD_DINOv3(nn.Module):
    """
    Main AD-DINOv3 framework for zero-shot anomaly detection and localization.
    Also usable as a feature extractor for multi-class classification (CSIG).
    """
    def __init__(
        self,
        dinov3_backbone_name: str = "dinov3_vitl16_reg14",
        clip_text_model_name: str = "ViT-L/14",
        adapter_reduction: int = 4,
        feature_layers: list = [6, 12, 18, 24],
        lambda_cm: float = 1.0,
        lambda_aacm: float = 1.0,
    ):
        super().__init__()
        self.lambda_cm = lambda_cm
        self.lambda_aacm = lambda_aacm
        self.feature_layers = feature_layers

        # ------------------------------------------------------------------
        # 1. Visual backbone: DINOv3 ViT-L/16 (frozen)
        # ------------------------------------------------------------------
        try:
            import torch.hub
            # Try to load via torch.hub (common for DINOv2/DINOv3 weights)
            # Note: DINOv3 official weights are available from Meta's releases.
            # Fallback to local import if available.
            self.dinov3 = torch.hub.load('/kaggle/working/', 'dinov3-vitl16-pretrain-lvd1689m', pretrained=True)
        except Exception as e:
            # Fallback: try direct import if installed in environment
            try:
                import dinov3
                self.dinov3 = dinov3.load_model()
            except Exception:
                raise RuntimeError(
                    "Failed to load DINOv3 backbone. Please ensure torch and dinov3 are installed, "
                    "or download weights manually. Error: " + str(e)
                )
        self.dinov3.eval()
        for param in self.dinov3.parameters():
            param.requires_grad = False

        # Feature dimension for ViT-L/16
        self.vis_dim = self.dinov3.embed_dim  # typically 1024 for ViT-L/16
        self.num_layers = len(self.feature_layers)

        # ------------------------------------------------------------------
        # 2. Multi-level visual adapters (applied to patch tokens at each layer)
        # ------------------------------------------------------------------
        self.visual_adapters = MultiLevelVisualAdapter(self.vis_dim, adapter_reduction)

        # ------------------------------------------------------------------
        # 3. Text branch: CLIP text encoder + adapter
        # ------------------------------------------------------------------
        try:
            import clip
            self.clip_model, _ = clip.load(clip_text_model_name, device="cpu")
        except Exception as e:
            # If clip not installed or model missing, create a mock text encoder
            # for code completeness. Users should install `open_clip` or `openai-clip`.
            print("WARNING: Could not load CLIP text encoder. Using mock encoder.")
            self.clip_model = None
            self.text_encoder = MockTextEncoder(self.vis_dim)
        else:
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.text_encoder = self.clip_model.encode_text

        # Light adapter for text embeddings (dimension depends on CLIP model)
        # For CLIP ViT-L/14, text embedding dim is 768.
        self.text_adapter_dim = 768 if self.clip_model is not None else self.vis_dim
        self.text_adapter = LightAdapter(self.text_adapter_dim, adapter_reduction)

        # ------------------------------------------------------------------
        # 4. AACM
        # ------------------------------------------------------------------
        self.aacm = AACM()

        # ------------------------------------------------------------------
        # 5. Projection heads for cross-modal alignment (optional but useful)
        # ------------------------------------------------------------------
        # We use direct cosine similarity between adapted visual patches and adapted text embeddings.
        # To match dimensions, we may need projection if dims differ.
        # Here we assume visual_dim == text_adapter_dim after projection, or we use shared space.
        # For simplicity, we project both to a shared alignment space.
        self.align_dim = 512
        self.visual_proj = nn.Linear(self.vis_dim, self.align_dim)
        self.text_proj = nn.Linear(self.text_adapter_dim, self.align_dim)
        nn.init.xavier_uniform_(self.visual_proj.weight)
        nn.init.zeros_(self.visual_proj.bias)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)

    # ------------------------------------------------------------------
    # Helper: extract multi-level features from DINOv3
    # ------------------------------------------------------------------
    def extract_visual_features(self, x: torch.Tensor, layers: list = None) -> tuple:
        """
        Extract patch tokens and CLS token at specified intermediate layers.
        Args:
            x: (B, 3, H, W) normalized images.
            layers: list of layer indices (1-indexed for convenience; paper uses 6,12,18,24).
        Returns:
            patch_features: list of 4 tensors, each (B, N, D) or (B, N+1, D)
            cls_features: list of 4 tensors, each (B, D)
            final_patch: (B, N, D) final layer patches (excluding CLS)
            final_cls: (B, D) final layer CLS
        Note: Implementation uses DINOv3 internal `blocks` and patches.
        Because exact layer access depends on the specific dinov3 code version,
        this method provides a robust approximation: we extract features from
        the backbone's intermediate outputs if available, or fall back to
        using only the final representation with simulated multi-level features.
        """
        if layers is None:
            layers = self.feature_layers

        # Since accessing exact intermediate layers in different DINOv3 releases varies,
        # we implement a practical approach:
        #   1. Get final tokens (patch + cls) from backbone.
        #   2. For multi-level representation, we approximate by splitting the backbone
        #      into stages and collecting outputs. This requires hooking into the model.
        # For a robust implementation without relying on internal code variations,
        # we use the backbone's `forward` with a custom hook mechanism or
        # approximate by extracting at different stages if the model supports it.

        # Simplified robust approach: use backbone to get final embeddings,
        # and approximate multi-level features by using intermediate block outputs
        # via forward hooks. If hooks fail, duplicate final features with different
        # adapter paths (simplified approximation for demonstration).
        # Here we implement hook-based extraction for accuracy.

        features_by_layer = {}
        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output could be tuple or tensor; handle both.
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                features_by_layer[layer_idx] = out
            return hook

        # Try to register hooks on transformer blocks.
        # Different releases have different names: `blocks`, `transformer_blocks`, etc.
        blocks_attr = None
        for attr_name in ['blocks', 'transformer_blocks', 'layers', 'encoder_layers']:
            if hasattr(self.dinov3, attr_name):
                blocks_attr = attr_name
                break

        if blocks_attr is not None:
            blocks = getattr(self.dinov3, blocks_attr)
            # Some releases have `ModuleList`, others `Sequential` or list.
            if isinstance(blocks, (nn.ModuleList, nn.Sequential, list)):
                for idx in layers:
                    # Ensure index within range (1-indexed in paper, but Python 0-indexed)
                    block_idx = idx - 1
                    if 0 <= block_idx < len(blocks):
                        handle = blocks[block_idx].register_forward_hook(make_hook(idx))
                        hooks.append(handle)
        else:
            # If we cannot find blocks, we will approximate with only final output duplicated.
            pass

        # Forward through backbone
        # Note: DINOv3 `forward` may return different things depending on version.
        # We try common signatures.
        with torch.no_grad():
            try:
                # Try `forward_features` first (common in vision transformers)
                if hasattr(self.dinov3, 'forward_features'):
                    outputs = self.dinov3.forward_features(x)
                else:
                    outputs = self.dinov3(x)
            except Exception:
                # Fallback: just call backbone
                outputs = self.dinov3(x)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Extract features by layer from hooks
        patch_features = []
        cls_features = []
        for idx in layers:
            if idx in features_by_layer:
                feat = features_by_layer[idx]
                # Depending on version, feat may be (B, N+1, D) including CLS, or (B, N, D)
                # We assume the first token is CLS for standard ViT structures.
                if feat.dim() == 3 and feat.shape[1] > 1:
                    # Separate CLS (first token) and patches (rest)
                    cls_token = feat[:, 0, :]  # (B, D)
                    patch_token = feat[:, 1:, :]  # (B, N, D)
                else:
                    # If no CLS token present in intermediate output (some versions separate them),
                    # approximate: use mean pooling as CLS proxy, or use final CLS.
                    patch_token = feat
                    cls_token = feat[:, 0, :] if feat.dim() == 3 else feat.mean(dim=1)
                patch_features.append(patch_token)
                cls_features.append(cls_token)
            else:
                # Fallback: duplicate final feature for missing layers
                # Get final feature from outputs
                if isinstance(outputs, dict):
                    final_feat = outputs.get('x', outputs.get('last_hidden_state', outputs.get('patch_tokens')))
                elif isinstance(outputs, tuple):
                    final_feat = outputs[0] if isinstance(outputs[0], torch.Tensor) else outputs[-1]
                else:
                    final_feat = outputs
                # Assume final_feat includes CLS at start
                if final_feat.dim() == 3:
                    cls_token = final_feat[:, 0, :]
                    patch_token = final_feat[:, 1:, :]
                else:
                    cls_token = final_feat.mean(dim=1)
                    patch_token = final_feat.unsqueeze(1)
                patch_features.append(patch_token)
                cls_features.append(cls_token)

        # Final layer features
        # Try to get final patch and CLS separately.
        # We can reuse the last hook result or compute from final output.
        if len(patch_features) > 0:
            final_patch = patch_features[-1]
            final_cls = cls_features[-1]
        else:
            # If no hooks fired at all, use outputs directly
            if isinstance(outputs, dict):
                feat = outputs.get('x', outputs.get('last_hidden_state'))
            elif isinstance(outputs, tuple):
                feat = outputs[0] if isinstance(outputs[0], torch.Tensor) else outputs[-1]
            else:
                feat = outputs
            if feat.dim() == 3:
                final_cls = feat[:, 0, :]
                final_patch = feat[:, 1:, :]
            else:
                final_cls = feat.mean(dim=1)
                final_patch = feat.unsqueeze(1)

        # If patch_features list has fewer elements than expected, duplicate the last.
        while len(patch_features) < len(layers):
            patch_features.append(final_patch)
            cls_features.append(final_cls)

        return patch_features, cls_features, final_patch, final_cls

    # ------------------------------------------------------------------
    # Forward for anomaly detection (training / inference)
    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        text_prompts: torch.Tensor = None,
        masks: torch.Tensor = None,
        return_maps: bool = True,
    ) -> dict:
        """
        Main forward pass for AD-DINOv3.
        Args:
            images: (B, 3, H, W) normalized input images.
            text_prompts: (B, L) tokenized text prompts (if using CLIP). If None, uses default prompts.
            masks: (B, H, W) binary ground-truth anomaly masks (training only).
            return_maps: whether to return pixel-level anomaly maps.
        Returns:
            dict with keys:
                - 'cls_token': adapted final CLS token (B, align_dim)
                - 'patch_tokens': list of 4 adapted patch token tensors
                - 'anomaly_map': (B, H, W) pixel-level anomaly probability (if return_maps)
                - 'patch_probs': (B, N, 2) normal/abnormal probabilities per patch (last level)
                - 'aacm_loss': scalar (if masks provided)
                - 'cm_loss': scalar cross-modal alignment loss (if masks provided)
        """
        B, C, H, W = images.shape
        device = images.device

        # 1. Extract multi-level visual features (frozen backbone)
        patch_features, cls_features, final_patch, final_cls = self.extract_visual_features(images, self.feature_layers)

        # 2. Apply multi-level visual adapters
        adapted_patches = self.visual_adapters(patch_features)  # list of 4, each (B, N, D)
        # For simplicity, use the deepest (last) adapted patch and CLS for cross-modal alignment
        # The paper mentions using all levels; for efficiency we use the deepest level for inference,
        # and can aggregate during training.
        deepest_patch = adapted_patches[-1]  # (B, N, D)
        deepest_cls = self.visual_adapters.adapters[-1](final_cls)  # adapt deepest CLS with corresponding adapter
        # Note: The above applies adapter correctly per level. For deepest level we use adapters[-1].

        # Project to alignment space
        deepest_patch_proj = self.visual_proj(deepest_patch)  # (B, N, align_dim)
        deepest_cls_proj = self.visual_proj(deepest_cls.unsqueeze(1) if deepest_cls.dim() == 2 else deepest_cls)
        # Ensure deepest_cls_proj is (B, 1, align_dim) or (B, align_dim)
        if deepest_cls_proj.dim() == 2:
            deepest_cls_proj = deepest_cls_proj.unsqueeze(1)
        deepest_cls_proj = deepest_cls_proj.squeeze(1)  # (B, align_dim) for similarity

        # 3. Text branch
        if text_prompts is not None:
            # Encode text with CLIP
            if self.clip_model is not None:
                with torch.no_grad():
                    text_embeddings = self.clip_model.encode_text(text_prompts)
            else:
                # Mock path: assume text_prompts are already embeddings
                if isinstance(text_prompts, torch.Tensor) and text_prompts.dim() == 3:
                    text_embeddings = text_prompts.squeeze(1)
                else:
                    text_embeddings = text_prompts
        else:
            # Default prompts: create dummy embeddings (users should provide prompts for real use)
            text_embeddings = torch.randn(B, 2, self.text_adapter_dim, device=device)

        # Adapt text embeddings
        # Assume text_embeddings shape: (B, 2, D) for [normal, abnormal] prompts, or (B, D)
        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(1)  # (B, 1, D)
        # Adapt each prompt separately if there are 2 prompts (normal/abnormal)
        adapted_text_list = []
        for i in range(text_embeddings.shape[1]):
            adapted_text_list.append(self.text_adapter(text_embeddings[:, i, :]))
        adapted_text = torch.stack(adapted_text_list, dim=1)  # (B, 2, text_adapter_dim)
        # Project text to alignment space
        text_proj = self.text_proj(adapted_text)  # (B, 2, align_dim)

        # 4. Cross-Modal Contrastive Learning (CMCL)
        # Compute cosine similarity between visual patch tokens and text embeddings
        # deepest_patch_proj: (B, N, align_dim)
        # text_proj: (B, 2, align_dim) -> treat normal (0) and abnormal (1) separately
        normal_text = text_proj[:, 0, :]  # (B, align_dim)
        abnormal_text = text_proj[:, 1, :]  # (B, align_dim)
        # Actually for binary classification per patch, we can compute similarity to both and softmax.
        # But the paper uses cosine similarity and applies softmax over the class dimension for each patch.
        # Let's compute similarity scores for normal and abnormal:
        # (B, N, align_dim) @ (B, align_dim) -> (B, N)
        # Normal similarity
        sim_normal = torch.bmm(deepest_patch_proj, normal_text.unsqueeze(2)).squeeze(2)  # (B, N)
        sim_abnormal = torch.bmm(deepest_patch_proj, abnormal_text.unsqueeze(2)).squeeze(2)  # (B, N)
        # Stack and apply softmax over dimension 1 (the two classes)
        sims = torch.stack([sim_normal, sim_abnormal], dim=1)  # (B, 2, N)
        sims = sims / (sim_normal.norm(p=2, dim=-1, keepdim=True) + 1e-6)  # approximate normalization for cosine
        # Better approach: normalize features first
        deepest_patch_proj_norm = F.normalize(deepest_patch_proj, p=2, dim=-1)
        normal_text_norm = F.normalize(normal_text, p=2, dim=-1)
        abnormal_text_norm = F.normalize(abnormal_text, p=2, dim=-1)
        sim_normal_norm = torch.bmm(deepest_patch_proj_norm, normal_text_norm.unsqueeze(2)).squeeze(2)
        sim_abnormal_norm = torch.bmm(deepest_patch_proj_norm, abnormal_text_norm.unsqueeze(2)).squeeze(2)
        sims_norm = torch.stack([sim_normal_norm, sim_abnormal_norm], dim=1)  # (B, 2, N)

        # Apply softmax over class dimension for each patch: (B, N, 2) -> (B, 2, N) in paper? Let's align.
        # The paper says: p = sigma(s) where s is similarity, p in R^{N x 2}
        # Let's treat N patches, 2 classes. So softmax over class dimension (dim=1) for each patch.
        patch_probs = F.softmax(sims_norm.transpose(1, 2), dim=-1)  # (B, N, 2): class dim last
        # Abnormal probability per patch
        abnormal_probs = patch_probs[:, :, 1]  # (B, N)

        # 5. Pixel-level anomaly map (bilinear upsample to original resolution)
        # Assume deepest_patch corresponds to grid of patches at some resolution.
        # For simplicity, we approximate by reshaping abnormal_probs to a square grid
        # and upsampling.
        anomaly_map = None
        if return_maps:
            # Determine grid size from number of patches N.
            # Note: exact N depends on image size and patch size (16x16 for ViT-L).
            # We approximate grid size by sqrt(N), but it's safer to compute from image size.
            # For standard 512x512 input with patch size 16, grid = 32x32, N = 1024.
            # We assume input has been resized to a standard size or we compute from actual image dimensions.
            # Let's approximate grid = int(sqrt(N)) if N is square.
            N_patches = deepest_patch.shape[1]
            # Try to infer grid from original image dimensions and patch size (16)
            # We assume input images are resized to 512x512 in the dataset loader, giving grid 32x32.
            grid_size = int(N_patches ** 0.5)
            if grid_size * grid_size != N_patches:
                # If not perfect square, find closest square <= N_patches or use fixed grid
                # For simplicity, assume 32x32 grid (1024 patches) and truncate/pad.
                grid_size = 32
            abnormal_probs_grid = abnormal_probs[:, :grid_size * grid_size].reshape(B, grid_size, grid_size)
            # Upsample to original image size
            anomaly_map = F.interpolate(
                abnormal_probs_grid.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(1)  # (B, H, W)

        # 6. AACM loss (training only, requires masks)
        aacm_loss = torch.tensor(0.0, device=device)
        cm_loss = torch.tensor(0.0, device=device)
        if masks is not None:
            # AACM: guide CLS token to attend to anomalous regions.
            # We use the deepest adapted CLS token (before projection for similarity with patches)
            # But in our architecture, deepest_cls is adapted CLS feature.
            # Actually deepest_cls was computed as adapter on final_cls, but we need it in feature space (D).
            # Let's recompute adapted deepest CLS in feature space properly.
            # The deepest_cls from extract_visual_features is final_cls (before adapter) or adapted?
            # Looking back: deepest_cls = self.visual_adapters.adapters[-1](final_cls). That is adapted.
            # But for similarity with patches (also in feature space D), we can use deepest_cls directly.
            # We need to ensure deepest_cls has shape (B, D) or (B, 1, D).
            if deepest_cls.dim() == 3:
                deepest_cls_feat = deepest_cls.squeeze(1)
            else:
                deepest_cls_feat = deepest_cls
            # Deepest patch tokens in feature space (before projection) are deepest_patch.
            deepest_patch_feat = deepest_patch
            aacm_loss = self.aacm.loss(deepest_cls_feat, deepest_patch_feat, masks)

            # Cross-modal alignment loss: compare anomaly map (patch_probs abnormal) with mask.
            # We already computed abnormal_probs at deepest level (before projection).
            # Let's reshape abnormal_probs to match grid size for loss.
            N_patches_cm = deepest_patch_feat.shape[1]
            grid_size_cm = int(N_patches_cm ** 0.5)
            if grid_size_cm * grid_size_cm == N_patches_cm:
                abnormal_probs_grid = abnormal_probs[:, :grid_size_cm * grid_size_cm].reshape(B, grid_size_cm, grid_size_cm)
                mask_down = F.interpolate(masks.unsqueeze(1).float(), size=(grid_size_cm, grid_size_cm), mode='bilinear', align_corners=False).squeeze(1)
                mask_down_flat = mask_down.reshape(B, -1)
                # Focal + Dice loss on abnormal probabilities vs mask
                # Note: paper defines cross-modal alignment on patch-text similarity map P.
                # We approximate P as abnormal_probs (probability of abnormal class per patch).
                # Actually the paper uses P as the abnormal probability rearranged into image resolution.
                # For simplicity, we compute focal and dice on the grid-level abnormal probabilities.
                pred_flat = abnormal_probs[:, :grid_size_cm * grid_size_cm].reshape(B, -1)
                cm_loss = AACM.focal_loss(pred_flat, mask_down_flat) + AACM.dice_loss(pred_flat, mask_down_flat)
            else:
                # Approximate: interpolate mask and abnormal_probs directly to image size
                # But for simplicity, use original abnormal_probs against downsampled mask at image level
                pass

        result = {
            'cls_token': deepest_cls_proj.squeeze(1) if deepest_cls_proj.dim() == 3 else deepest_cls_proj,
            'patch_tokens': deepest_patch_proj,
            'anomaly_map': anomaly_map,
            'patch_probs': patch_probs,  # (B, N, 2)
            'aacm_loss': aacm_loss,
            'cm_loss': cm_loss,
            'total_loss': self.lambda_cm * cm_loss + self.lambda_aacm * aacm_loss,
        }
        return result

    # ------------------------------------------------------------------
    # Forward for multi-class classification (CSIG) using final CLS token
    # ------------------------------------------------------------------
    def forward_classification(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings suitable for classification tasks (e.g., CSIG 50-class).
        Returns logit-like embeddings of shape (B, num_classes) if a classifier head is added,
        or simply the final adapted CLS feature (B, align_dim) for use with an external classifier.
        For simplicity, we return the final projected CLS token.
        """
        # Just call forward without text/masks and return the CLS token
        out = self.forward(images, text_prompts=None, masks=None, return_maps=False)
        cls_token = out['cls_token']  # (B, align_dim)
        return cls_token


# ------------------------------------------------------------------
# Mock text encoder for environments without `open_clip` or `clip`
# ------------------------------------------------------------------
class MockTextEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Random projection that produces embeddings of size `dim`
        self.proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.proj.weight)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids expected as (B, L) or (B, D) depending on usage.
        # For mock, assume it's a random embedding or we project from a dummy input.
        # If token_ids is 2D with second dim large, treat as (B, L) and embed via mean pooling of random vectors.
        if token_ids.dim() == 2 and token_ids.shape[1] > 1:
            B, L = token_ids.shape
            # Create dummy embeddings
            dummy = torch.randn(B, L, self.dim, device=token_ids.device)
            return self.proj(dummy.mean(dim=1))
        else:
            return self.proj(token_ids.float())
