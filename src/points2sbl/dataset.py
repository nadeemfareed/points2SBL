# src/points2sbl/dataset.py
from __future__ import annotations

import os
import glob
from typing import Dict, Any, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return a / np.maximum(b, eps)


def _sanitize_pos(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    if not np.isfinite(pos).all():
        pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    return pos


def _eig_from_neighbors(pts: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    pts: (N,3)
    returns:
      w_desc: (N,3) eigenvalues sorted desc (l1>=l2>=l3), clamped >=0
      v_desc: (N,3,3) eigenvectors aligned with w_desc (columns correspond to l1,l2,l3)
    """
    pts = np.asarray(pts, dtype=np.float64)
    N = pts.shape[0]
    if N < 3:
        w = np.zeros((N, 3), dtype=np.float64)
        v = np.zeros((N, 3, 3), dtype=np.float64)
        v[:] = np.eye(3, dtype=np.float64)
        return w, v

    k_eff = int(min(max(3, k), max(3, N - 1)))

    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k_eff, workers=-1)  # (N,k)
    neigh = pts[idx]                                # (N,k,3)

    mu = neigh.mean(axis=1, keepdims=True)
    X = neigh - mu                                  # (N,k,3)
    C = np.einsum("nki,nkj->nij", X, X) / max(1, (k_eff - 1))  # (N,3,3)

    w_asc, v_asc = np.linalg.eigh(C)  # ascending
    w_asc = np.maximum(w_asc, 0.0)

    w_desc = w_asc[:, ::-1]
    v_desc = v_asc[:, :, ::-1]
    return w_desc, v_desc


def _geom_features_from_eigs(w_desc: np.ndarray, select: Sequence[str]) -> np.ndarray:
    w_desc = np.asarray(w_desc, dtype=np.float64)
    l1, l2, l3 = w_desc[:, 0], w_desc[:, 1], w_desc[:, 2]
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
        elif n == "anisotropy":
            out.append(_safe_div(l1 - l3, l1))
        elif n == "sum_eigs":
            out.append(s)
        elif n == "omnivariance":
            out.append(np.power(np.maximum(l1 * l2 * l3, 0.0), 1.0 / 3.0))
        elif n == "omnivariance_log":
            omni = np.power(np.maximum(l1 * l2 * l3, 0.0), 1.0 / 3.0)
            out.append(np.log(np.maximum(omni, 1e-12)))
        elif n == "eigenentropy":
            p1 = _safe_div(l1, s)
            p2 = _safe_div(l2, s)
            p3 = _safe_div(l3, s)
            ent = -(p1 * np.log(np.maximum(p1, 1e-12)) +
                    p2 * np.log(np.maximum(p2, 1e-12)) +
                    p3 * np.log(np.maximum(p3, 1e-12)))
            out.append(ent)
        elif n == "l1":
            out.append(l1)
        elif n == "l2":
            out.append(l2)
        elif n == "l3":
            out.append(l3)
        else:
            raise ValueError(f"Unknown geom feature: {name}")

    X = np.stack(out, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


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


class BlockNPYDataset(Dataset):
    """
    Supports:
      - *.npz with keys: 'pos' (N,3), 'y' (N,)
      - *.npy raw arrays or dicts (kept for backward compatibility)

    ALWAYS returns:
      pos: (N,3) float32
      x  : (N,F) float32  (computed from pos)
      y  : (N,)  int64
    """

    def __init__(self, sample_dir: str, feature_cfg: Dict[str, Any], num_classes: int, train_mode: bool = True):
        self.sample_dir = str(sample_dir)
        self.feature_cfg = dict(feature_cfg or {})
        self.num_classes = int(num_classes)
        self.train_mode = bool(train_mode)

        self.files = sorted(glob.glob(os.path.join(self.sample_dir, "*.npz")))
        self.file_kind = "npz"
        if not self.files:
            self.files = sorted(glob.glob(os.path.join(self.sample_dir, "*.npy")))
            self.file_kind = "npy"

        if not self.files:
            raise FileNotFoundError(f"No .npz/.npy blocks found in: {self.sample_dir}")

        self.base_seed = int(self.feature_cfg.get("seed", 42))
        self.clip_features = float(self.feature_cfg.get("clip_features", 5.0))
        self.standardize = bool(self.feature_cfg.get("standardize_features", True))

    def __len__(self) -> int:
        return len(self.files)

    def _load_block(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if path.lower().endswith(".npz"):
            z = np.load(path)
            if "pos" not in z.files or "y" not in z.files:
                raise KeyError(f"NPZ missing required keys pos/y: {path} keys={z.files}")
            pos = z["pos"]
            y = z["y"]
            return pos, y

        # NPY legacy paths
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            try:
                arr = arr.item()
            except Exception:
                pass

        if isinstance(arr, dict):
            pos = arr.get("pos", None)
            y = arr.get("y", None)
            if pos is None or y is None:
                raise KeyError(f"NPY dict missing pos/y: {path} keys={list(arr.keys())}")
            return pos, y

        if hasattr(arr, "ndim") and arr.ndim == 2 and arr.shape[1] >= 4:
            pos = arr[:, :3]
            y = arr[:, -1]
            return pos, y

        raise ValueError(f"Unsupported NPY block format: {path}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[int(idx)]
        pos, y = self._load_block(path)

        pos = _sanitize_pos(pos)  # (N,3)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        y = np.clip(y, 0, self.num_classes - 1)

        rng = np.random.RandomState((self.base_seed * 1000003 + int(idx)) & 0xFFFFFFFF)

        feats: List[np.ndarray] = []

        # xyz as features (recommended for PointNeXt)
        if bool(self.feature_cfg.get("use_xyz", True)):
            if bool(self.feature_cfg.get("use_centered_xyz", False)):
                feats.append((pos - pos.mean(axis=0, keepdims=True)).astype(np.float32))
            else:
                feats.append(pos.astype(np.float32))

        # geom features from eigs
        if bool(self.feature_cfg.get("include_geom_features", False)):
            k_geom = _choose_k(self.feature_cfg, self.train_mode, rng, "geom_k", "geom_k_infer")
            w_desc, _ = _eig_from_neighbors(pos, k=k_geom)

            select = self.feature_cfg.get("geom_select", ["linearity", "planarity", "scattering"])
            if not isinstance(select, (list, tuple)) or len(select) == 0:
                select = ["linearity", "planarity", "scattering"]

            g = _geom_features_from_eigs(w_desc, select=select)
            feats.append(g)

        if len(feats) == 0:
            x = np.zeros((pos.shape[0], 1), dtype=np.float32)
        else:
            x = np.concatenate(feats, axis=1).astype(np.float32, copy=False)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.standardize and x.shape[1] > 0:
            x = _standardize_clip(x, clip=self.clip_features)

        return {
            "pos": torch.from_numpy(pos.astype(np.float32, copy=False)),
            "x": torch.from_numpy(x.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y),
        }
