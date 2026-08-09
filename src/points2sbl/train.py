#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/points2sbl/train.py (ROBUST SUPERVISED TRAINING)

Revised metrics:
- Supports block formats:
    *.npz with keys ['pos','y']          (NO x saved)
    *.npz with keys ['pos','x','y']
    legacy *.npy blocks (N,>=4) with last col = y
- If x missing, computes features ON THE FLY from pos using YAML features:
    xyz/centered xyz + geometric features
- Correct OneCycleLR stepping: step() ONCE per optimizer step
- Robust printing: heartbeat every N optimizer steps, plus val progress
- AMP support using torch.amp, auto-disables on overflow (optional)
- SSL ckpt is optional and OFF if null/none/empty

Added evaluation metrics:
- Overall Accuracy (OA)
- Precision / Recall / F1 for woody and leaf
- IoU for woody and leaf
- Macro F1
- Balanced Accuracy
- mIoU
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .models import build_model
from .losses import CombinedLoss, LossConfig


# ---------------------------
# YAML + seeding
# ---------------------------

def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _now() -> float:
    return time.time()


def _safe_int(x, default=None):
    if x in (None, "", "0", 0):
        return default
    return int(x)


def _gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0.0


# ---------------------------
# Feature computation (ON THE FLY when x not stored)
# ---------------------------

def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return a / np.maximum(b, eps)


def _standardize_clip(X: np.ndarray, clip: float = 5.0) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-6)
    Z = (X - mu) / sd
    Z = np.clip(Z, -float(clip), float(clip))
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z


def _choose_k(feature_cfg: Dict[str, Any], train_mode: bool, rng: np.random.RandomState, key: str, key_infer: str) -> int:
    v = feature_cfg.get(key, 24)
    v_infer = int(feature_cfg.get(key_infer, 24))
    if isinstance(v, (list, tuple)):
        ks = [int(k) for k in v if int(k) >= 3]
        if not ks:
            return v_infer
        if train_mode:
            return int(rng.choice(ks))
        return int(v_infer if v_infer in ks else ks[len(ks) // 2])
    return int(v)


def _eig_from_neighbors(pts: np.ndarray, k: int) -> np.ndarray:
    """
    pts: (N,3)
    returns w_desc: (N,3) with l1>=l2>=l3
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype=np.float64)
    N = pts.shape[0]
    if N < 3:
        return np.zeros((N, 3), dtype=np.float64)

    k_eff = int(min(max(3, k), max(3, N - 1)))

    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k_eff, workers=1)
    neigh = pts[idx]

    mu = neigh.mean(axis=1, keepdims=True)
    X = neigh - mu
    C = np.einsum("nki,nkj->nij", X, X) / max(1, (k_eff - 1))

    w_asc, _ = np.linalg.eigh(C)
    w_asc = np.maximum(w_asc, 0.0)
    w_desc = w_asc[:, ::-1]
    return w_desc


def _geom_features_from_eigs(w_desc: np.ndarray, select: List[str]) -> np.ndarray:
    w_desc = np.asarray(w_desc, dtype=np.float64)
    l1 = w_desc[:, 0]
    l2 = w_desc[:, 1]
    l3 = w_desc[:, 2]
    s = l1 + l2 + l3

    out: List[np.ndarray] = []
    for name in select:
        n = str(name).lower().strip()
        if n == "linearity":
            out.append(_safe_div(l1 - l2, l1))
        elif n == "planarity":
            out.append(_safe_div(l2 - l3, l1))
        elif n == "scattering":
            out.append(_safe_div(l3, l1))
        elif n == "curvature":
            out.append(_safe_div(l3, s))
        else:
            raise ValueError(f"Unknown geom feature: {name}")

    X = np.stack(out, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def compute_features_from_pos(
    pos: np.ndarray,
    feature_cfg: Dict[str, Any],
    train_mode: bool,
    rng: np.random.RandomState,
) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    if not np.isfinite(pos).all():
        pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)

    feats: List[np.ndarray] = []

    use_xyz = bool(feature_cfg.get("use_xyz", True))
    use_centered = bool(feature_cfg.get("use_centered_xyz", False))
    if use_xyz:
        if use_centered:
            feats.append((pos - pos.mean(axis=0, keepdims=True)).astype(np.float32))
        else:
            feats.append(pos.astype(np.float32))

    if bool(feature_cfg.get("include_geom_features", False)):
        k_geom = _choose_k(feature_cfg, train_mode, rng, "geom_k", "geom_k_infer")
        w_desc = _eig_from_neighbors(pos, k=k_geom)
        select = feature_cfg.get("geom_select", ["linearity", "planarity", "scattering"])
        if not isinstance(select, (list, tuple)) or len(select) == 0:
            select = ["linearity", "planarity", "scattering"]
        feats.append(_geom_features_from_eigs(w_desc, list(select)))

    if len(feats) == 0:
        x = np.zeros((pos.shape[0], 1), dtype=np.float32)
    else:
        x = np.concatenate(feats, axis=1).astype(np.float32, copy=False)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if bool(feature_cfg.get("standardize_features", True)):
        clip = float(feature_cfg.get("clip_features", 5.0))
        x = _standardize_clip(x, clip=clip)

    return x


# ---------------------------
# Dataset: npz(pos,y) / npz(pos,x,y) / npy legacy
# ---------------------------

class BlockDataset(Dataset):
    def __init__(self, sample_dir: str, feature_cfg: Dict[str, Any], num_classes: int, train_mode: bool):
        import glob

        self.sample_dir = str(sample_dir)
        self.feature_cfg = dict(feature_cfg or {})
        self.num_classes = int(num_classes)
        self.train_mode = bool(train_mode)
        self.base_seed = int(self.feature_cfg.get("seed", 42))

        npz = sorted(glob.glob(os.path.join(self.sample_dir, "*.npz")))
        npy = sorted(glob.glob(os.path.join(self.sample_dir, "*.npy")))
        self.files = npz + npy

        if not self.files:
            raise FileNotFoundError(f"No blocks found in: {self.sample_dir} (expected .npz or .npy)")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[int(idx)]
        ext = os.path.splitext(path)[1].lower()

        rng = np.random.RandomState((self.base_seed * 1000003 + int(idx)) & 0xFFFFFFFF)

        if ext == ".npz":
            z = np.load(path)
            pos = z["pos"].astype(np.float32, copy=False)
            y = z["y"].astype(np.int64, copy=False)
            x = z["x"].astype(np.float32, copy=False) if ("x" in z.files) else None
            if x is None:
                x = compute_features_from_pos(pos, self.feature_cfg, self.train_mode, rng)
        else:
            arr = np.load(path)
            arr = arr.astype(np.float32, copy=False)
            if arr.ndim != 2 or arr.shape[1] < 4:
                raise ValueError(f"Bad legacy .npy block: {path} shape={arr.shape}")
            pos = arr[:, :3].astype(np.float32, copy=False)
            y = arr[:, -1].astype(np.int64, copy=False)
            x = compute_features_from_pos(pos, self.feature_cfg, self.train_mode, rng)

        if not np.isfinite(pos).all():
            pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        y = np.clip(y, 0, self.num_classes - 1)

        return {
            "pos": torch.from_numpy(pos),
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
        }


# ---------------------------
# Loss + metrics
# ---------------------------

def make_loss(cfg: Dict[str, Any], num_classes: int) -> CombinedLoss:
    tr = cfg.get("train", {}) or {}
    lc = (tr.get("loss", {}) or {})

    loss_cfg = LossConfig(
        ce_w=float(lc.get("ce_w", 1.0)),
        dice_w=float(lc.get("dice_w", 0.0)),
        focal_w=float(lc.get("focal_w", 0.0)),
        focal_gamma=float(lc.get("focal_gamma", 2.0)),
        label_smoothing=float(lc.get("label_smoothing", 0.0)),
        ignore_index=int(lc.get("ignore_index", -1)),
        ce_class_weights=lc.get("ce_class_weights", None),
    )
    return CombinedLoss(loss_cfg, num_classes=num_classes)


def confusion_2class(pred: torch.Tensor, y: torch.Tensor) -> Tuple[int, int, int, int]:
    pred = pred.view(-1)
    y = y.view(-1)
    tp = int(((pred == 1) & (y == 1)).sum().item())
    tn = int(((pred == 0) & (y == 0)).sum().item())
    fp = int(((pred == 1) & (y == 0)).sum().item())
    fn = int(((pred == 0) & (y == 1)).sum().item())
    return tn, fp, fn, tp


@torch.no_grad()
def metrics_from_confusion(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    total = tn + fp + fn + tp
    oa = (tp + tn) / max(1, total)

    # class 1 = leaf
    leaf_p = tp / max(1, tp + fp)
    leaf_r = tp / max(1, tp + fn)
    leaf_f1 = 0.0 if (leaf_p + leaf_r) == 0 else 2.0 * leaf_p * leaf_r / (leaf_p + leaf_r)
    leaf_iou = tp / max(1, tp + fp + fn)

    # class 0 = woody
    woody_p = tn / max(1, tn + fn)
    woody_r = tn / max(1, tn + fp)
    woody_f1 = 0.0 if (woody_p + woody_r) == 0 else 2.0 * woody_p * woody_r / (woody_p + woody_r)
    woody_iou = tn / max(1, tn + fp + fn)

    macro_f1 = 0.5 * (woody_f1 + leaf_f1)
    balanced_acc = 0.5 * (woody_r + leaf_r)
    miou = 0.5 * (woody_iou + leaf_iou)

    return {
        "oa": float(oa),
        "acc": float(oa),  # backward compatibility
        "leaf_p": float(leaf_p),
        "leaf_r": float(leaf_r),
        "leaf_f1": float(leaf_f1),
        "leaf_iou": float(leaf_iou),
        "woody_p": float(woody_p),
        "woody_r": float(woody_r),
        "woody_f1": float(woody_f1),
        "woody_iou": float(woody_iou),
        "macro_f1": float(macro_f1),
        "balanced_acc": float(balanced_acc),
        "miou": float(miou),
    }


# ---------------------------
# Train / eval loops
# ---------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: CombinedLoss,
    device: torch.device,
    use_amp_cfg: bool,
    grad_clip: Optional[float],
    ignore_index: int,
    max_steps: Optional[int],
    log_every: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    steps = 0
    skip = 0

    amp_enabled = bool(use_amp_cfg and device.type == "cuda")
    t_last = _now()

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        pos = batch["pos"].to(device, non_blocking=True).float()
        x = batch["x"].to(device, non_blocking=True).float()
        y = batch["y"].to(device, non_blocking=True).long()

        valid = (y != ignore_index)
        if not valid.any():
            skip += 1
            continue

        optimizer.zero_grad(set_to_none=True)

        try:
            if amp_enabled:
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(pos, x)
                    if not torch.isfinite(logits).all():
                        skip += 1
                        continue
                    loss = loss_fn(logits, y)
                if not torch.isfinite(loss):
                    skip += 1
                    continue

                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                if steps < 5:
                    found_bad = False
                    for p in model.parameters():
                        if p.grad is not None and (not torch.isfinite(p.grad).all()):
                            found_bad = True
                            break
                    if found_bad:
                        print(f"[WARN] AMP overflow at step={step}. Disabling AMP and continuing.", flush=True)
                        amp_enabled = False

            else:
                logits = model(pos, x)
                if not torch.isfinite(logits).all():
                    skip += 1
                    continue
                loss = loss_fn(logits, y)
                if not torch.isfinite(loss):
                    skip += 1
                    continue

                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

        except RuntimeError as e:
            skip += 1
            if steps < 3:
                print(f"[SKIP] RuntimeError at step={step}: {type(e).__name__}: {e}", flush=True)
            continue

        total_loss += float(loss.item())
        steps += 1

        if steps == 1:
            print(f"[INFO] first optimizer step OK | loss={float(loss.item()):.6f}", flush=True)

        if log_every and (steps % int(log_every) == 0):
            dt = _now() - t_last
            t_last = _now()
            avg = total_loss / max(1, steps)
            mem = _gpu_mem_mb()
            lr_now = float(optimizer.param_groups[0]["lr"])
            print(
                f"train[{steps}] step={step+1}/{len(loader)} "
                f"loss={avg:.6f} skip={skip} amp={'ON' if amp_enabled else 'OFF'} "
                f"lr={lr_now:.6g} dt={dt:.1f}s gpu_mem={mem:.0f}MB",
                flush=True,
            )

    return {
        "train_loss": float(total_loss / max(1, steps)),
        "train_steps": float(steps),
        "train_skip": float(skip),
        "amp_used": 1.0 if amp_enabled else 0.0,
    }


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: CombinedLoss,
    device: torch.device,
    ignore_index: int,
    max_steps: Optional[int],
    log_every: int,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    steps = 0
    skip = 0

    tn = fp = fn = tp = 0
    total_points = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        pos = batch["pos"].to(device, non_blocking=True).float()
        x = batch["x"].to(device, non_blocking=True).float()
        y = batch["y"].to(device, non_blocking=True).long()

        valid = (y != ignore_index)
        if not valid.any():
            skip += 1
            continue

        logits = model(pos, x)
        if not torch.isfinite(logits).all():
            skip += 1
            continue

        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            skip += 1
            continue

        total_loss += float(loss.item())
        steps += 1

        pred = logits.argmax(dim=1)
        yv = torch.clamp(y[valid], 0, 1)
        pv = torch.clamp(pred[valid], 0, 1)

        total_points += int(yv.numel())
        tni, fpi, fni, tpi = confusion_2class(pv, yv)
        tn += tni
        fp += fpi
        fn += fni
        tp += tpi

        if log_every and (steps % int(log_every) == 0):
            print(
                f"val[{steps}] step={step+1}/{len(loader)} "
                f"loss={total_loss / max(1, steps):.6f} skip={skip}",
                flush=True,
            )

    m = metrics_from_confusion(tn, fp, fn, tp)
    m.update(
        {
            "val_loss": float(total_loss / max(1, steps)),
            "val_steps": float(steps),
            "val_skip": float(skip),
            "val_points": float(total_points),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
        }
    )
    return m


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    data_cfg = cfg.get("data", {}) or {}
    train_dir = data_cfg.get("train_dir", "train")
    val_dir = data_cfg.get("val_dir", "val")

    train_sample = Path(args.data_root) / train_dir / "sample_dir"
    val_sample = Path(args.data_root) / val_dir / "sample_dir"
    if not train_sample.exists():
        raise FileNotFoundError(f"train sample_dir not found: {train_sample}")
    if not val_sample.exists():
        raise FileNotFoundError(f"val sample_dir not found: {val_sample}")

    feat_cfg = cfg.get("features", {}) or {}
    tr_cfg = cfg.get("train", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    num_classes = int(model_cfg.get("num_classes", 2))
    model_cfg["num_classes"] = num_classes

    seed = int(tr_cfg.get("seed", 42))
    epochs = int(tr_cfg.get("epochs", 40))
    batch_size = int(tr_cfg.get("batch_size", 8))
    lr = float(tr_cfg.get("lr", 1e-4))
    weight_decay = float(tr_cfg.get("weight_decay", 0.01))
    num_workers = int(tr_cfg.get("num_workers", 0))

    device = torch.device(tr_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    use_amp_cfg = bool(tr_cfg.get("mixed_precision", False))
    grad_clip = float(tr_cfg.get("grad_clip", 0.0)) if tr_cfg.get("grad_clip", 0.0) else None

    ignore_index = int(((tr_cfg.get("loss", {}) or {}).get("ignore_index", -1)))

    max_steps = _safe_int(tr_cfg.get("max_steps_per_epoch", None), default=None)
    val_max_steps = _safe_int(tr_cfg.get("val_max_steps", None), default=None)
    val_every = int(tr_cfg.get("val_every", 1))
    log_every = int(tr_cfg.get("progress_every", 50))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(seed)

    print(f"[INFO] device={device.type} | torch={torch.__version__} | cuda_avail={torch.cuda.is_available()}", flush=True)

    train_ds = BlockDataset(str(train_sample), feat_cfg, num_classes=num_classes, train_mode=True)
    val_ds = BlockDataset(str(val_sample), feat_cfg, num_classes=num_classes, train_mode=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    b0 = next(iter(train_loader))
    x0 = b0.get("x", None)
    in_dim = int(x0.shape[-1]) if x0 is not None else 3
    model_cfg["in_dim"] = in_dim

    model = build_model(model_cfg, in_dim).to(device)

    ssl_ckpt = (tr_cfg.get("ssl_ckpt", None) or "")
    ssl_ckpt = str(ssl_ckpt).strip()
    if ssl_ckpt and ssl_ckpt.lower() not in ("none", "null", "false", "0"):
        p = Path(ssl_ckpt)
        if p.exists():
            ck = torch.load(str(p), map_location="cpu")
            state = ck.get("model", ck.get("state_dict", ck if isinstance(ck, dict) else ck))
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[INFO] loaded ssl_ckpt: {p}", flush=True)
            print(f"[INFO] ssl missing keys: {len(missing)}", flush=True)
            print(f"[INFO] ssl unexpected keys: {len(unexpected)}", flush=True)
        else:
            print(f"[WARN] ssl_ckpt not found: {p}", flush=True)
    else:
        print("[INFO] ssl_ckpt disabled (supervised).", flush=True)

    y0 = b0["y"].reshape(-1).cpu().numpy()
    u, c = np.unique(y0, return_counts=True)
    dist = dict(zip([int(v) for v in u.tolist()], [int(v) for v in c.tolist()]))

    print(f"[INFO] inferred in_dim={in_dim} (x present=True) | num_classes={num_classes}", flush=True)
    print(f"[INFO] batch0 label dist: {dist} (ignore_index={ignore_index})", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    sch = (tr_cfg.get("scheduler", {}) or {})
    if str(sch.get("name", "")).lower() == "one_cycle":
        max_lr = float(sch.get("max_lr", lr * 3.0))
        pct_start = float(sch.get("pct_start", 0.1))
        div_factor = float(sch.get("div_factor", 10.0))
        final_div_factor = float(sch.get("final_div_factor", 200.0))

        steps_per_epoch = len(train_loader) if max_steps is None else min(len(train_loader), int(max_steps))
        total_steps = max(1, steps_per_epoch * epochs)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy="cos",
        )
        print(f"[INFO] OneCycleLR enabled | total_steps={total_steps} max_lr={max_lr}", flush=True)

    use_amp = bool(use_amp_cfg and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    loss_fn = make_loss(cfg, num_classes=num_classes).to(device)

    with open(out_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_macro = -1.0

    for epoch in range(1, epochs + 1):
        t0 = _now()
        print(f"\n===== EPOCH {epoch}/{epochs} =====", flush=True)

        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            use_amp_cfg=use_amp,
            grad_clip=grad_clip,
            ignore_index=ignore_index,
            max_steps=max_steps,
            log_every=max(1, log_every),
            scheduler=scheduler,
        )

        metrics: Dict[str, float] = {"epoch": float(epoch), **tr}

        if (epoch % val_every) == 0:
            va = eval_one_epoch(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                ignore_index=ignore_index,
                max_steps=val_max_steps,
                log_every=max(0, log_every // 2),
            )
            metrics.update(va)

            dt = _now() - t0
            print(
                f"[E{epoch:03d}] "
                f"train_loss={metrics['train_loss']:.6f} | "
                f"val_loss={metrics['val_loss']:.6f} | "
                f"OA={metrics['oa']:.4f} | "
                f"woody(P/R/F1/IoU)={metrics['woody_p']:.3f}/{metrics['woody_r']:.3f}/{metrics['woody_f1']:.3f}/{metrics['woody_iou']:.3f} | "
                f"leaf(P/R/F1/IoU)={metrics['leaf_p']:.3f}/{metrics['leaf_r']:.3f}/{metrics['leaf_f1']:.3f}/{metrics['leaf_iou']:.3f} | "
                f"macroF1={metrics['macro_f1']:.3f} | "
                f"mIoU={metrics['miou']:.3f} | "
                f"balAcc={metrics['balanced_acc']:.3f} | "
                f"train_skip={int(metrics['train_skip'])} | val_skip={int(metrics['val_skip'])} | "
                f"amp={'ON' if int(metrics.get('amp_used', 0)) == 1 else 'OFF'} | "
                f"{dt:.1f}s",
                flush=True,
            )

            if int(metrics["val_steps"]) == 0:
                print(
                    "[WARN] Validation produced 0 steps (all batches skipped). "
                    "This usually means nonfinite pos/x, y all ignore_index, or val loader empty.",
                    flush=True,
                )

            if metrics["macro_f1"] > best_macro:
                best_macro = float(metrics["macro_f1"])
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": cfg,
                        "best_macro_f1": best_macro,
                        "metrics": metrics,
                    },
                    out_dir / "best.pt",
                )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": cfg,
                "best_macro_f1": best_macro,
                "metrics": metrics,
            },
            out_dir / "last.pt",
        )

        with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    print(f"\nDone. best_macro_f1={best_macro:.4f}", flush=True)
    print(f"Saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()