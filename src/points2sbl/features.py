# src/points2sbl/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ----------------------------
# KNN helpers (same behavior as before, but centralized)
# ----------------------------
def _safe_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (n + eps)


def _try_knn_indices(xyz: np.ndarray, k: int) -> np.ndarray:
    """
    Prefer sklearn if available, else brute force.
    Returns (N,k) neighbor indices including self.
    """
    N = xyz.shape[0]
    k = int(min(max(k, 3), N))
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(xyz)
        idx = nn.kneighbors(xyz, return_distance=False)
        return idx.astype(np.int64)
    except Exception:
        # brute force fallback (OK for blocks; slower for huge N)
        # compute squared distances
        d2 = np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=2)
        idx = np.argsort(d2, axis=1)[:, :k]
        return idx.astype(np.int64)


GEOM_EIGEN_NAMES = [
    "linearity",
    "planarity",
    "scattering",
    "omnivariance",
    "anisotropy",
    "eigenentropy",
    "curvature",
]


def compute_normals_and_eigen_features(xyz: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      normals: (N,3)
      eigen_feats: (N,7) in GEOM_EIGEN_NAMES order
    """
    xyz = xyz.astype(np.float32, copy=False)
    N = xyz.shape[0]
    k = int(min(max(k, 3), N))

    idx = _try_knn_indices(xyz, k=k)  # (N,k)
    nbr = xyz[idx]  # (N,k,3)

    mu = np.mean(nbr, axis=1, keepdims=True)
    X = nbr - mu
    cov = np.einsum("nki,nkj->nij", X, X) / float(idx.shape[1])

    w, v = np.linalg.eigh(cov)  # ascending
    w = np.clip(w, 1e-12, None)

    # normal = eigenvector of smallest eigenvalue
    normals = _safe_unit(v[:, :, 0].astype(np.float32))

    # sort eigenvalues desc for features
    w_desc = w[:, ::-1]
    l1, l2, l3 = w_desc[:, 0], w_desc[:, 1], w_desc[:, 2]
    s = l1 + l2 + l3

    linearity = (l1 - l2) / (l1 + 1e-12)
    planarity = (l2 - l3) / (l1 + 1e-12)
    scattering = l3 / (l1 + 1e-12)
    omnivariance = np.cbrt(np.clip(l1 * l2 * l3, 1e-36, None))
    anisotropy = (l1 - l3) / (l1 + 1e-12)
    p1, p2, p3 = l1 / (s + 1e-12), l2 / (s + 1e-12), l3 / (s + 1e-12)
    eigenentropy = -(p1 * np.log(p1 + 1e-12) + p2 * np.log(p2 + 1e-12) + p3 * np.log(p3 + 1e-12))
    curvature = l3 / (s + 1e-12)

    eigen_feats = np.stack(
        [linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy, curvature],
        axis=1,
    ).astype(np.float32)

    return normals, eigen_feats


@dataclass
class FeatureConfig:
    use_xyz: bool = True
    use_centered_xyz: bool = False

    include_geom_features: bool = False
    geom_select: Optional[List[str]] = None

    # NEW: variable K support
    geom_k_list: Optional[List[int]] = None   # e.g. [12,24,48]
    geom_k_infer: int = 24                    # used for val/test/predict

    # legacy toggles (keep for backward compatibility)
    use_normals: bool = False
    normal_k: int = 24
    use_eigen: bool = False
    eigen_k: int = 24

    normalize_features: bool = True
    standardize_features: bool = True
    clip_features: float = 5.0
    log_omnivariance: bool = False

    def choose_geom_k(self, train: bool, rng: np.random.Generator) -> int:
        """
        Training: randomly pick from geom_k_list if provided, else geom_k_infer.
        Val/Test/Predict: geom_k_infer.
        """
        if (not train) or (not self.geom_k_list) or (len(self.geom_k_list) == 0):
            return int(self.geom_k_infer)
        ks = [int(k) for k in self.geom_k_list if int(k) >= 3]
        if not ks:
            return int(self.geom_k_infer)
        return int(rng.choice(ks))


def feature_dim_from_cfg(fcfg: FeatureConfig) -> int:
    d = 0
    if fcfg.use_xyz:
        d += 3
    if fcfg.use_centered_xyz:
        d += 3

    if fcfg.include_geom_features and fcfg.geom_select:
        d += int(len(fcfg.geom_select))
    else:
        # legacy full geometry
        if fcfg.use_normals:
            d += 3
        if fcfg.use_eigen:
            d += 7
    return d


def build_features_xyz_geom(
    xyz: np.ndarray,
    fcfg: FeatureConfig,
    train: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns x features (N,F) matching your YAML settings.
    IMPORTANT: Feature dimension is independent of K when using variable-K.
    """
    xyz = xyz.astype(np.float32, copy=False)
    feats: List[np.ndarray] = []

    if fcfg.use_xyz:
        feats.append(xyz)

    if fcfg.use_centered_xyz:
        feats.append(xyz - xyz.mean(axis=0, keepdims=True))

    if fcfg.include_geom_features and fcfg.geom_select:
        k = fcfg.choose_geom_k(train=train, rng=rng)
        _, eigen = compute_normals_and_eigen_features(xyz, k=k)

        # select requested channels
        name_to_i = {n: i for i, n in enumerate(GEOM_EIGEN_NAMES)}
        cols = []
        for n in fcfg.geom_select:
            if n not in name_to_i:
                raise ValueError(f"Unknown geom feature '{n}'. Options: {GEOM_EIGEN_NAMES}")
            cols.append(eigen[:, name_to_i[n]])

        geom = np.stack(cols, axis=1).astype(np.float32)

        # optional transform
        if fcfg.log_omnivariance and "omnivariance" in (fcfg.geom_select or []):
            j = (fcfg.geom_select or []).index("omnivariance")
            geom[:, j] = np.log(np.clip(geom[:, j], 1e-12, None)).astype(np.float32)

        feats.append(geom)

    x = np.concatenate(feats, axis=1).astype(np.float32) if feats else xyz.astype(np.float32)

    # Stabilization (applied per block)
    if fcfg.normalize_features:
        # normalize each channel to roughly [-1,1] using robust scaling by std
        mu = np.mean(x, axis=0, keepdims=True)
        sd = np.std(x, axis=0, keepdims=True) + 1e-6
        x = (x - mu) / sd

    if fcfg.standardize_features:
        mu = np.mean(x, axis=0, keepdims=True)
        sd = np.std(x, axis=0, keepdims=True) + 1e-6
        x = (x - mu) / sd

    if fcfg.clip_features and float(fcfg.clip_features) > 0:
        c = float(fcfg.clip_features)
        x = np.clip(x, -c, c)

    return x.astype(np.float32)
