from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utilities
# =============================================================================

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Pairwise squared distances.

    src: (B, N, 3)
    dst: (B, M, 3)
    returns: (B, N, M)
    """
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return torch.clamp(dist, min=0.0)


def knn_idx(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """
    k-NN indices in xyz space.

    xyz: (B, N, 3)
    returns: (B, N, k)
    """
    B, N, _ = xyz.shape
    k = min(k, N)
    dist = square_distance(xyz, xyz)  # (B,N,N)
    idx = dist.topk(k=k, dim=-1, largest=False, sorted=False)[1]
    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather points by index.

    points: (B, N, C)
    idx:    (B, S) or (B, S, K)
    returns: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


# =============================================================================
# Transformer Block
# =============================================================================

class PTBlock(nn.Module):
    def __init__(self, dim: int, k: int, dropout: float):
        super().__init__()
        self.k = k

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(p=dropout)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dim * 4, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        xyz: torch.Tensor,
        x: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        rel_xyz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        xyz:     (B, N, 3)
        x:       (B, N, D)
        idx:     optional shared xyz-kNN indices (B, N, k)
        rel_xyz: optional shared relative xyz (B, N, k, 3)

        All Point Transformer blocks operate on the same xyz coordinates.
        The spatial kNN graph can therefore be calculated once per forward
        pass and reused without changing neighborhood topology or weights.
        """
        B, N, D = x.shape

        if idx is None:
            idx = knn_idx(xyz, self.k)

        if rel_xyz is None:
            knn_xyz = index_points(xyz, idx)
            rel_xyz = knn_xyz - xyz.unsqueeze(2)

        knn_x = index_points(x, idx)       # (B,N,k,D)

        q = self.to_q(x).unsqueeze(2)      # (B,N,1,D)
        k = self.to_k(knn_x)               # (B,N,k,D)
        v = self.to_v(knn_x)               # (B,N,k,D)

        pos = self.pos_mlp(rel_xyz)            # (B,N,k,D)

        attn = self.attn_mlp(q - k + pos)      # (B,N,k,D)
        attn = F.softmax(attn, dim=2)          # softmax over k

        out = torch.sum(attn * (v + pos), dim=2)  # (B,N,D)
        out = self.drop(self.proj(out))

        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


# =============================================================================
# Config and Segmentation Model
# =============================================================================

@dataclass
class PointTransformerConfig:
    num_classes: int = 3
    in_dim: int = 3
    dim: int = 128
    depth: int = 6
    knn_k: int = 24
    dropout: float = 0.1


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class PointTransformerBackbone(nn.Module):
    """
    Backbone that outputs per-point embeddings (no classifier head).
    Useful for self-supervised pretraining.
    """
    def __init__(self, cfg: PointTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.dim, cfg.dim),
        )

        self.blocks = nn.ModuleList(
            [PTBlock(cfg.dim, cfg.knn_k, cfg.dropout) for _ in range(cfg.depth)]
        )

        self.apply(_init_weights)

    def forward(self, pos: torch.Tensor, feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        pos:  (B, N, 3)
        feat: (B, N, Fin) or None (if None, xyz used as features)
        returns: (B, N, dim)
        """
        if feat is None:
            feat = pos
        x = self.in_proj(feat)  # (B,N,dim)

        # xyz stays fixed throughout the encoder.
        idx = knn_idx(pos, self.cfg.knn_k)
        rel_xyz = index_points(pos, idx) - pos.unsqueeze(2)

        for blk in self.blocks:
            x = blk(
                pos,
                x,
                idx=idx,
                rel_xyz=rel_xyz,
            )

        return x


class PointTransformerSeg(nn.Module):
    """
    Point-wise segmentation head on top of PointTransformer encoder.
    """
    def __init__(self, cfg: PointTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.dim, cfg.dim),
        )

        self.blocks = nn.ModuleList(
            [PTBlock(cfg.dim, cfg.knn_k, cfg.dropout) for _ in range(cfg.depth)]
        )

        self.head = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.dim, cfg.num_classes),
        )

        self.apply(_init_weights)

    def forward(self, pos: torch.Tensor, feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        pos:  (B, N, 3)
        feat: (B, N, Fin) or None (if None, xyz used as features)
        returns: logits (B, C, N)
        """
        if feat is None:
            feat = pos
        x = self.in_proj(feat)  # (B,N,dim)

        # Calculate xyz-only neighborhood once instead of once per PT block.
        idx = knn_idx(pos, self.cfg.knn_k)
        rel_xyz = index_points(pos, idx) - pos.unsqueeze(2)

        for blk in self.blocks:
            x = blk(
                pos,
                x,
                idx=idx,
                rel_xyz=rel_xyz,
            )

        logits = self.head(x)   # (B,N,C)
        return logits.permute(0, 2, 1).contiguous()  # (B,C,N)
