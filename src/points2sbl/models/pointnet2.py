from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# =============================================================================
# Utilities
# =============================================================================

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared distances.

    src: (B, N, 3)
    dst: (B, M, 3)
    returns: (B, N, M)
    """
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))  # (B,N,M)
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)   # (B,N,1)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)    # (B,1,M)
    return torch.clamp(dist, min=0.0)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B, N, C)
    idx:    (B, S) or (B, S, K)
    return: (B, S, C) or (B, S, K, C)
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


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest point sampling.

    xyz: (B, N, 3)
    npoint: number of points to sample (clamped to N)
    returns: (B, npoint) indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    npoint = min(npoint, N)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    k-NN search in xyz.

    xyz:     (B, N, 3)
    new_xyz: (B, S, 3)
    returns: (B, S, k) indices
    """
    B, N, _ = xyz.shape
    k = min(k, N)
    dist = square_distance(new_xyz, xyz)  # (B,S,N)
    idx = dist.topk(k=k, dim=-1, largest=False, sorted=False)[1]  # (B,S,k)
    return idx


def sample_and_group(
    npoint: int,
    k: int,
    xyz: torch.Tensor,
    points: Optional[torch.Tensor],
):
    """
    Farthest point sample + kNN grouping.

    xyz:    (B, N, 3)
    points: (B, N, C) or None
    returns:
      new_xyz:   (B, S, 3)
      new_points:(B, S, k, 3+C)
    """
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)               # (B,S,3)
    idx = knn_point(k, xyz, new_xyz)                   # (B,S,k)
    grouped_xyz = index_points(xyz, idx)               # (B,S,k,3)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

    if points is not None:
        grouped_points = index_points(points, idx)     # (B,S,k,C)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


def interpolate_features(
    xyz1: torch.Tensor, xyz2: torch.Tensor, points2: torch.Tensor, k: int = 3
) -> torch.Tensor:
    """
    Feature interpolation from xyz2 to xyz1.

    xyz1:    (B, N, 3)
    xyz2:    (B, S, 3)
    points2: (B, S, C2)
    returns: (B, N, C2)
    """
    B, N, _ = xyz1.shape
    S = xyz2.shape[1]
    k = min(k, S)

    dist = square_distance(xyz1, xyz2)  # (B,N,S)
    d, idx = dist.topk(k=k, dim=-1, largest=False, sorted=False)  # (B,N,k)
    d = torch.clamp(d, min=1e-10)

    w = 1.0 / d
    w = w / torch.sum(w, dim=-1, keepdim=True)  # (B,N,k)

    interpolated = torch.sum(index_points(points2, idx) * w.unsqueeze(-1), dim=2)
    return interpolated


# =============================================================================
# Blocks
# =============================================================================

class SharedMLP2d(nn.Module):
    def __init__(self, in_ch: int, out_chs: Tuple[int, ...]):
        super().__init__()
        layers = []
        last = in_ch
        for oc in out_chs:
            layers += [
                nn.Conv2d(last, oc, 1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            ]
            last = oc
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedMLP1d(nn.Module):
    def __init__(self, in_ch: int, out_chs: Tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        layers = []
        last = in_ch
        for oc in out_chs:
            layers += [
                nn.Conv1d(last, oc, 1, bias=False),
                nn.BatchNorm1d(oc),
                nn.ReLU(inplace=True),
            ]
            last = oc
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetAbstraction(nn.Module):
    def __init__(self, npoint: int, k: int, in_ch: int, mlp: Tuple[int, ...]):
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.mlp = SharedMLP2d(in_ch, mlp)

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]):
        new_xyz, new_points = sample_and_group(self.npoint, self.k, xyz, points)
        # new_points: (B,S,k,C)
        new_points = new_points.permute(0, 3, 1, 2).contiguous()  # (B,C,S,k)
        new_points = self.mlp(new_points)                         # (B,C,S,k)
        new_points = torch.max(new_points, dim=-1)[0]             # (B,C,S)
        return new_xyz, new_points.transpose(1, 2).contiguous()   # (B,S,C)


class FeaturePropagation(nn.Module):
    def __init__(self, in_ch: int, mlp: Tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        self.mlp = SharedMLP1d(in_ch, mlp, dropout=dropout)

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: Optional[torch.Tensor],
        points2: torch.Tensor,
    ):
        """
        xyz1:    (B,N,3)   target
        xyz2:    (B,S,3)   source
        points1: (B,N,C1)  features at xyz1 or None
        points2: (B,S,C2)  features at xyz2
        """
        interpolated = interpolate_features(xyz1, xyz2, points2, k=3)  # (B,N,C2)
        if points1 is not None:
            new_points = torch.cat([interpolated, points1], dim=-1)    # (B,N,C1+C2)
        else:
            new_points = interpolated
        new_points = new_points.transpose(1, 2).contiguous()           # (B,C,N)
        new_points = self.mlp(new_points)                              # (B,C,N)
        return new_points.transpose(1, 2).contiguous()                 # (B,N,C)


# =============================================================================
# PointNet++ Segmentation
# =============================================================================

@dataclass
class PointNet2Config:
    num_classes: int = 3
    input_dim: int = 3
    sa1_npoint: int = 2048
    sa1_k: int = 32
    sa2_npoint: int = 512
    sa2_k: int = 32
    sa3_npoint: int = 128
    sa3_k: int = 32
    dropout: float = 0.2
    feat_dropout: float = 0.0  # extra feature dropout before head


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class PointNet2Seg(nn.Module):
    def __init__(self, cfg: PointNet2Config):
        super().__init__()
        self.cfg = cfg

        # Encoder (set abstraction)
        self.sa1 = SetAbstraction(
            cfg.sa1_npoint, cfg.sa1_k, in_ch=3 + cfg.input_dim, mlp=(64, 64, 128)
        )
        self.sa2 = SetAbstraction(
            cfg.sa2_npoint, cfg.sa2_k, in_ch=3 + 128, mlp=(128, 128, 256)
        )
        self.sa3 = SetAbstraction(
            cfg.sa3_npoint, cfg.sa3_k, in_ch=3 + 256, mlp=(256, 256, 512)
        )

        # Decoder (feature propagation)
        self.fp3 = FeaturePropagation(
            in_ch=512 + 256, mlp=(256, 256), dropout=cfg.dropout
        )
        self.fp2 = FeaturePropagation(
            in_ch=256 + 128, mlp=(256, 128), dropout=cfg.dropout
        )
        self.fp1 = FeaturePropagation(
            in_ch=128 + cfg.input_dim, mlp=(128, 128, 128), dropout=cfg.dropout
        )

        # Optional feature dropout before head
        self.feat_drop = nn.Dropout1d(p=cfg.feat_dropout) if cfg.feat_dropout > 0 else nn.Identity()

        # Segmentation head
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
            nn.Conv1d(128, cfg.num_classes, 1),
        )

        # Initialize weights
        self.apply(_init_weights)

    def forward(
        self,
        pos: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pos:  (B,N,3)
        feat: (B,N,Fin) or None (if None, xyz used as features)
        returns: logits (B, C, N)
        """
        if feat is None:
            feat = pos

        l0_xyz = pos
        l0_points = feat

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)     # (B,N1,3), (B,N1,128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)     # (B,N2,3), (B,N2,256)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)     # (B,N3,3), (B,N3,512)

        l2_points_up = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # (B,N2,256)
        l1_points_up = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_up)  # (B,N1,128+?)
        l0_points_up = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points_up)  # (B,N,128)

        x = l0_points_up.transpose(1, 2).contiguous()  # (B,128,N)
        x = self.feat_drop(x)
        return self.head(x)  # (B,C,N)
