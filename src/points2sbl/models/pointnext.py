# src/points2sbl/models/pointnext.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Utils
# =============================================================================

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    # src: (B, N, 3), dst: (B, M, 3) -> (B, N, M)
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return torch.clamp(dist, min=0.0)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # points: (B, N, C), idx: (B, S) or (B, S, K)
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    # xyz: (B, N, 3) -> (B, npoint)
    device = xyz.device
    B, N, _ = xyz.shape
    npoint = min(int(npoint), int(N))

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
    # xyz: (B, N, 3), new_xyz: (B, S, 3) -> (B, S, k)
    B, N, _ = xyz.shape
    k = min(int(k), int(N))
    dist = square_distance(new_xyz, xyz)  # (B,S,N)
    idx = dist.topk(k=k, dim=-1, largest=False, sorted=False)[1]
    return idx


def interpolate_features(
    xyz1: torch.Tensor, xyz2: torch.Tensor, points2: torch.Tensor, k: int = 3
) -> torch.Tensor:
    # xyz1: (B, N, 3), xyz2: (B, S, 3), points2: (B, S, C2) -> (B, N, C2)
    B, N, _ = xyz1.shape
    S = xyz2.shape[1]
    k = min(int(k), int(S))

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

def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class PreMLP(nn.Module):
    # (B,N,C) -> (B,N,Cout)
    def __init__(self, in_ch: int, out_ch: int, act: str = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, out_ch, bias=False)
        self.ln1 = nn.LayerNorm(out_ch)
        self.fc2 = nn.Linear(out_ch, out_ch, bias=False)
        self.ln2 = nn.LayerNorm(out_ch)
        self.act = _act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        return x


class LocalAgg(nn.Module):
    # grouped: (B, S, K, Cin) + geo -> pool -> (B, Cout, S)
    def __init__(self, in_ch: int, out_ch: int, act: str = "relu", use_dist: bool = True):
        super().__init__()
        self.use_dist = bool(use_dist)
        geo_ch = 3 + (1 if self.use_dist else 0)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch + geo_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            _act(act),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            _act(act),
        )

    def forward(self, new_xyz: torch.Tensor, grouped_xyz: torch.Tensor, grouped_feat: torch.Tensor) -> torch.Tensor:
        xyz_rel = grouped_xyz - new_xyz.unsqueeze(2)  # (B,S,K,3)
        if self.use_dist:
            dist = torch.norm(xyz_rel, dim=-1, keepdim=True)  # (B,S,K,1)
            geo = torch.cat([xyz_rel, dist], dim=-1)          # (B,S,K,4)
        else:
            geo = xyz_rel                                     # (B,S,K,3)

        x = torch.cat([geo, grouped_feat], dim=-1)            # (B,S,K,geo+Cin)
        x = x.permute(0, 3, 1, 2).contiguous()                # (B,C,S,K)
        x = self.mlp(x)                                       # (B,Cout,S,K)
        x = torch.max(x, dim=-1)[0]                           # (B,Cout,S)
        return x


class InvResMLP(nn.Module):
    def __init__(self, ch: int, expand_ratio: int = 4, dropout: float = 0.0, act: str = "relu"):
        super().__init__()
        mid = int(ch) * int(expand_ratio)
        self.net = nn.Sequential(
            nn.Conv1d(ch, mid, 1, bias=False),
            nn.BatchNorm1d(mid),
            _act(act),
            nn.Conv1d(mid, ch, 1, bias=False),
            nn.BatchNorm1d(ch),
            nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity(),
        )
        self.act = _act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SetAbstractionPointNeXt(nn.Module):
    def __init__(
        self,
        npoint: int,
        k: int,
        in_ch: int,
        out_ch: int,
        blocks_per_stage: int = 2,
        expand_ratio: int = 4,
        act: str = "relu",
        dropout: float = 0.0,
        use_dist: bool = True,
        use_pre_mlp: bool = True,
    ):
        super().__init__()
        self.npoint = int(npoint)
        self.k = int(k)

        self.use_pre_mlp = bool(use_pre_mlp)
        self.pre = PreMLP(in_ch, out_ch, act=act) if self.use_pre_mlp else None
        self.local = LocalAgg(out_ch if self.use_pre_mlp else in_ch, out_ch, act=act, use_dist=use_dist)

        blocks = []
        for _ in range(int(blocks_per_stage)):
            blocks.append(InvResMLP(out_ch, expand_ratio=expand_ratio, dropout=dropout, act=act))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        # xyz: (B,N,3), feat: (B,N,Cin)
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)          # (B,S,3)

        if self.pre is not None:
            feat = self.pre(feat)                     # (B,N,out_ch)

        idx = knn_point(self.k, xyz, new_xyz)         # (B,S,K)
        grouped_xyz = index_points(xyz, idx)          # (B,S,K,3)
        grouped_feat = index_points(feat, idx)        # (B,S,K,C)

        x = self.local(new_xyz, grouped_xyz, grouped_feat)   # (B,out,S)
        x = self.blocks(x)                                    # (B,out,S)
        return new_xyz, x.transpose(1, 2).contiguous()        # (B,S,out)


class FeaturePropagationPointNeXt(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, act: str = "relu"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            _act(act),
            nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            _act(act),
        )

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, feat1: Optional[torch.Tensor], feat2: torch.Tensor):
        # xyz1: (B,N,3) target, xyz2: (B,S,3) source
        # feat1: (B,N,C1) or None, feat2: (B,S,C2)
        interpolated = interpolate_features(xyz1, xyz2, feat2, k=3)  # (B,N,C2)
        if feat1 is not None:
            new_feat = torch.cat([interpolated, feat1], dim=-1)      # (B,N,C1+C2)
        else:
            new_feat = interpolated
        new_feat = new_feat.transpose(1, 2).contiguous()             # (B,C,N)
        return self.mlp(new_feat).transpose(1, 2).contiguous()       # (B,N,out)


# =============================================================================
# Config + Model (EXPORTS PointNeXtConfig)
# =============================================================================

@dataclass
class PointNeXtConfig:
    num_classes: int = 2
    in_dim: int = 3

    sa1_npoint: int = 2048
    sa1_k: int = 32
    sa2_npoint: int = 512
    sa2_k: int = 32
    sa3_npoint: int = 128
    sa3_k: int = 32

    width1: int = 64
    width2: int = 128
    width3: int = 256

    # PointNeXt-ish knobs
    blocks_per_stage: int = 2
    expand_ratio: int = 4
    act: str = "relu"
    dropout: float = 0.2
    use_dist: bool = True
    use_pre_mlp: bool = True

    # kept for compatibility with your YAML keys
    pn2_dropout: float = 0.2
    feat_dropout: float = 0.0


class PointNeXtSeg(nn.Module):
    def __init__(self, cfg: PointNeXtConfig):
        super().__init__()
        self.cfg = cfg

        self.sa1 = SetAbstractionPointNeXt(
            npoint=cfg.sa1_npoint, k=cfg.sa1_k,
            in_ch=cfg.in_dim, out_ch=cfg.width1,
            blocks_per_stage=cfg.blocks_per_stage,
            expand_ratio=cfg.expand_ratio,
            act=cfg.act,
            dropout=cfg.dropout * 0.25,
            use_dist=cfg.use_dist,
            use_pre_mlp=cfg.use_pre_mlp,
        )
        self.sa2 = SetAbstractionPointNeXt(
            npoint=cfg.sa2_npoint, k=cfg.sa2_k,
            in_ch=cfg.width1, out_ch=cfg.width2,
            blocks_per_stage=cfg.blocks_per_stage,
            expand_ratio=cfg.expand_ratio,
            act=cfg.act,
            dropout=cfg.dropout * 0.25,
            use_dist=cfg.use_dist,
            use_pre_mlp=True,
        )
        self.sa3 = SetAbstractionPointNeXt(
            npoint=cfg.sa3_npoint, k=cfg.sa3_k,
            in_ch=cfg.width2, out_ch=cfg.width3,
            blocks_per_stage=cfg.blocks_per_stage,
            expand_ratio=cfg.expand_ratio,
            act=cfg.act,
            dropout=cfg.dropout * 0.25,
            use_dist=cfg.use_dist,
            use_pre_mlp=True,
        )

        self.fp3 = FeaturePropagationPointNeXt(cfg.width3 + cfg.width2, cfg.width2, dropout=cfg.dropout * 0.25, act=cfg.act)
        self.fp2 = FeaturePropagationPointNeXt(cfg.width2 + cfg.width1, cfg.width1, dropout=cfg.dropout * 0.25, act=cfg.act)
        self.fp1 = FeaturePropagationPointNeXt(cfg.width1 + cfg.in_dim, cfg.width1, dropout=cfg.dropout * 0.25, act=cfg.act)

        self.feat_drop = nn.Dropout1d(p=float(cfg.feat_dropout)) if cfg.feat_dropout > 0 else nn.Identity()

        self.head = nn.Sequential(
            nn.Conv1d(cfg.width1, cfg.width1, 1, bias=False),
            nn.BatchNorm1d(cfg.width1),
            _act(cfg.act),
            nn.Dropout(p=float(cfg.dropout)) if cfg.dropout > 0 else nn.Identity(),
            nn.Conv1d(cfg.width1, cfg.num_classes, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, pos: torch.Tensor, feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pos: (B,N,3), feat: (B,N,in_dim) or None
        if feat is None:
            feat = pos
            if self.cfg.in_dim != 3:
                raise ValueError(f"feat=None implies in_dim=3, but cfg.in_dim={self.cfg.in_dim}. Pass feat explicitly.")

        l0_xyz, l0_feat = pos, feat
        l1_xyz, l1_feat = self.sa1(l0_xyz, l0_feat)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)

        l2_up = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat)
        l1_up = self.fp2(l1_xyz, l2_xyz, l1_feat, l2_up)
        l0_up = self.fp1(l0_xyz, l1_xyz, l0_feat, l1_up)

        x = l0_up.transpose(1, 2).contiguous()
        x = self.feat_drop(x)
        return self.head(x)  # (B,C,N)


# Backward compatibility if other code imports this name
PointNeXtSegConfig = PointNeXtConfig
