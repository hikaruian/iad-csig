"""INP-Former (CVPR 2025) + optional INP-Former++ residual."""

from __future__ import annotations

import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint

from .blocks import AggregationBlock, Mlp, PrototypeBlock
from .encoder import DinoV2Encoder


TRAINABLE_PREFIXES = ("bottleneck.", "aggregation.", "decoder.", "prototype_token")


class INPFormer(nn.Module):
    def __init__(
        self,
        encoder: DinoV2Encoder,
        inp_num: int = 6,
        decoder_depth: int = 8,
        bottleneck_drop: float = 0.0,
        fuse_layer_encoder: Optional[List[List[int]]] = None,
        fuse_layer_decoder: Optional[List[List[int]]] = None,
        residual: bool = False,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = encoder.embed_dim
        self.num_heads = encoder.num_heads
        self.target_layers = encoder.target_layers
        self.inp_num = inp_num
        self.residual = residual
        self.grad_checkpoint = grad_checkpoint
        self.fuse_layer_encoder = fuse_layer_encoder or [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.fuse_layer_decoder = fuse_layer_decoder or [[0, 1, 2, 3], [4, 5, 6, 7]]

        dim = self.embed_dim
        heads = self.num_heads
        norm = partial(nn.LayerNorm, eps=1e-8)

        self.bottleneck = nn.ModuleList([Mlp(dim, dim * 4, dim, drop=bottleneck_drop)])
        self.prototype_token = nn.Parameter(torch.randn(inp_num, dim))
        self.aggregation = nn.ModuleList(
            [AggregationBlock(dim=dim, num_heads=heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=norm)]
        )
        self.decoder = nn.ModuleList(
            [
                PrototypeBlock(dim=dim, num_heads=heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=norm)
                for _ in range(decoder_depth)
            ]
        )
        self._init_trainable()

    def _init_trainable(self):
        for m in list(self.bottleneck.modules()) + list(self.aggregation.modules()) + list(self.decoder.modules()):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        trunc_normal_(self.prototype_token, std=0.02)

    def trainable_parameters(self):
        for p in (
            list(self.bottleneck.parameters())
            + list(self.aggregation.parameters())
            + list(self.decoder.parameters())
            + [self.prototype_token]
        ):
            yield p

    def trainable_state_dict(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if k.startswith(TRAINABLE_PREFIXES) or k == "prototype_token"}

    @staticmethod
    def fuse_feature(feat_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def gather_loss(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        dist = 1.0 - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        distance, _ = torch.min(dist, dim=2)
        return distance.mean()

    def encode_tokens(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        en_list = self.encoder.forward_features(x, self.target_layers)
        n_tok = en_list[0].shape[1]
        side = int(math.sqrt(n_tok))
        if side * side != n_tok:
            raise RuntimeError(
                f"Token count {n_tok} is not a square — check that image_size is divisible by patch size "
                f"({self.encoder.patch_size})."
            )
        return en_list, side

    def _run_block(self, blk, tokens, proto):
        if self.grad_checkpoint and self.training:
            try:
                return checkpoint(blk, tokens, proto, use_reentrant=False)
            except TypeError:
                return checkpoint(blk, tokens, proto)
        return blk(tokens, proto)

    def reconstruct(self, en_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        b = en_list[0].shape[0]
        fused = self.fuse_feature(en_list)
        proto = self.prototype_token.unsqueeze(0).expand(b, -1, -1)
        for blk in self.aggregation:
            proto = self._run_block(blk, proto, fused)
        g_loss = self.gather_loss(fused, proto)

        tokens = fused
        for blk in self.bottleneck:
            tokens = blk(tokens)

        de_list = []
        for blk in self.decoder:
            tokens = self._run_block(blk, tokens, proto)
            de_list.append(tokens)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[i] for i in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[i] for i in idxs]) for idxs in self.fuse_layer_decoder]
        if self.residual:
            de = [e.detach() + d for d, e in zip(de, en)]
        return en, de, g_loss

    def forward(self, x: torch.Tensor):
        en_list, side = self.encode_tokens(x)
        en, de, g_loss = self.reconstruct(en_list)
        b = en[0].shape[0]
        en = [e.permute(0, 2, 1).reshape(b, -1, side, side).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape(b, -1, side, side).contiguous() for d in de]
        return en, de, g_loss

    @torch.no_grad()
    def predict(self, x: torch.Tensor, out_size: int = 448) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        en, de, _ = self.forward(x)
        amap = anomaly_map_from_features(en, de, out_size=out_size)
        score = topk_score(amap, max_ratio=0.01)
        return amap, score


def anomaly_map_from_features(en, de, out_size: int = 448) -> torch.Tensor:
    maps = []
    for e, d in zip(en, de):
        a = 1.0 - F.cosine_similarity(e, d, dim=1)
        a = a.unsqueeze(1)
        a = F.interpolate(a, size=(out_size, out_size), mode="bilinear", align_corners=True)
        maps.append(a)
    return torch.cat(maps, dim=1).mean(dim=1, keepdim=True)


def topk_score(amap: torch.Tensor, max_ratio: float = 0.01) -> torch.Tensor:
    flat = amap.flatten(1)
    k = max(1, int(flat.shape[1] * max_ratio))
    return torch.topk(flat, k, dim=1).values.mean(dim=1)


def build_model(
    encoder_name: str = "dinov2reg_vit_base_14",
    inp_num: int = 6,
    decoder_depth: int = 8,
    bottleneck_drop: float = 0.0,
    residual: bool = False,
    encoder_source: str = "auto",
    grad_checkpoint: bool = False,
) -> INPFormer:
    encoder = DinoV2Encoder(encoder_name, source=encoder_source)
    return INPFormer(
        encoder=encoder,
        inp_num=inp_num,
        decoder_depth=decoder_depth,
        bottleneck_drop=bottleneck_drop,
        residual=residual,
        grad_checkpoint=grad_checkpoint,
    )
