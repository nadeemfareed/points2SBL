from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    ce_w: float = 1.0
    dice_w: float = 0.0
    focal_w: float = 0.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    ignore_index: int = -1

    # Optional class weights for CE, e.g. [2.0, 1.0] to penalize class-0 errors more.
    # Only used if provided.
    ce_class_weights: Optional[Sequence[float]] = None


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = -1, eps: float = 1e-6):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, N)
        target: (B, N) int
        """
        if logits.ndim != 3:
            raise ValueError(f"DiceLoss expects logits (B,C,N). Got {tuple(logits.shape)}")
        if target.ndim != 2:
            raise ValueError(f"DiceLoss expects target (B,N). Got {tuple(target.shape)}")

        B, C, N = logits.shape
        if C != self.num_classes:
            raise ValueError(f"DiceLoss num_classes mismatch: logits C={C} vs {self.num_classes}")

        valid = (target != self.ignore_index)
        if not valid.any():
            # no valid points -> return 0 to avoid NaNs, caller may skip anyway
            return logits.new_tensor(0.0)

        # one-hot
        with torch.no_grad():
            tgt = target.clamp(min=0, max=C - 1)
            onehot = F.one_hot(tgt, num_classes=C).permute(0, 2, 1).float()  # (B,C,N)
            onehot = onehot * valid.unsqueeze(1).float()

        probs = torch.softmax(logits, dim=1) * valid.unsqueeze(1).float()

        inter = (probs * onehot).sum(dim=2)  # (B,C)
        union = (probs + onehot).sum(dim=2)  # (B,C)
        dice = (2.0 * inter + self.eps) / (union + self.eps)  # (B,C)

        loss = 1.0 - dice.mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, N)
        target: (B, N)
        """
        if logits.ndim != 3:
            raise ValueError(f"FocalLoss expects logits (B,C,N). Got {tuple(logits.shape)}")
        if target.ndim != 2:
            raise ValueError(f"FocalLoss expects target (B,N). Got {tuple(target.shape)}")

        B, C, N = logits.shape
        valid = (target != self.ignore_index)
        if not valid.any():
            return logits.new_tensor(0.0)

        # flatten valid
        logits_v = logits.permute(0, 2, 1)[valid]  # (M, C)
        target_v = target[valid]                  # (M,)

        logp = F.log_softmax(logits_v, dim=1)
        p = logp.exp()
        pt = p.gather(1, target_v.view(-1, 1)).squeeze(1)
        loss = -((1.0 - pt) ** self.gamma) * logp.gather(1, target_v.view(-1, 1)).squeeze(1)
        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, cfg: LossConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(num_classes)

        # Build CE weights tensor if provided
        self.register_buffer("_ce_weight", None, persistent=False)
        if cfg.ce_class_weights is not None:
            w = torch.tensor(list(cfg.ce_class_weights), dtype=torch.float32)
            if w.numel() != self.num_classes:
                raise ValueError(
                    f"ce_class_weights length {w.numel()} must equal num_classes {self.num_classes}"
                )
            self._ce_weight = w

        self.dice = DiceLoss(num_classes=self.num_classes, ignore_index=cfg.ignore_index)
        self.focal = FocalLoss(gamma=cfg.focal_gamma, ignore_index=cfg.ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, N)
        target: (B, N)
        """
        if logits.ndim != 3:
            raise ValueError(f"CombinedLoss expects logits (B,C,N). Got {tuple(logits.shape)}")
        if target.ndim != 2:
            raise ValueError(f"CombinedLoss expects target (B,N). Got {tuple(target.shape)}")

        # CE expects (B,C,N) and (B,N)
        ce = F.cross_entropy(
            logits,
            target,
            ignore_index=self.cfg.ignore_index,
            label_smoothing=float(self.cfg.label_smoothing),
            weight=self._ce_weight,
        )

        loss = self.cfg.ce_w * ce

        if self.cfg.dice_w and self.cfg.dice_w > 0:
            loss = loss + self.cfg.dice_w * self.dice(logits, target)

        if self.cfg.focal_w and self.cfg.focal_w > 0:
            loss = loss + self.cfg.focal_w * self.focal(logits, target)

        return loss
