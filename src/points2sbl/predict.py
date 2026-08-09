from .model_manager import download_model
from __future__ import annotations

import argparse
import os
import sys
import time as _time
import hashlib
import json
from pathlib import Path
from importlib import resources
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch

from .utils import load_yaml
from .io_las import read_las, write_las_with_classification
from .models import build_model


def _get(d: dict, path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur



def _bundled_point_transformer_config() -> str:
    """Resolve the packaged Point Transformer YAML."""
    try:
        ref = resources.files("points2sbl").joinpath(
            "configs", "point_transformer.yaml"
        )
        return str(ref)
    except Exception:
        return str(
            Path(__file__).resolve().parent
            / "configs"
            / "point_transformer.yaml"
        )


def _default_checkpoint_candidates() -> List[Path]:
    """Ordered checkpoint locations used when --ckpt is omitted."""
    candidates: List[Path] = []

    env_ckpt = os.environ.get("POINTS2SBL_CKPT")
    if env_ckpt:
        candidates.append(Path(env_ckpt).expanduser())

    here = Path(__file__).resolve()

    try:
        repo_root = here.parents[2]
        candidates.append(
            repo_root
            / "runs"
            / "point_transformer_curated_20260327_170108"
            / "best.pt"
        )
    except Exception:
        pass

    candidates.append(
        Path.cwd()
        / "runs"
        / "point_transformer_curated_20260327_170108"
        / "best.pt"
    )

    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        cache_root = Path(os.environ["LOCALAPPDATA"]) / "points2sbl" / "models"
    else:
        cache_root = Path.home() / ".cache" / "points2sbl" / "models"

    candidates.extend(
        [
            cache_root / "point_transformer_best.pt",
            cache_root / "best.pt",
        ]
    )

    unique: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _resolve_checkpoint_path(requested: Optional[str]) -> str:
    """Resolve an explicit or default Point Transformer checkpoint."""
    if requested:
        p = Path(requested).expanduser()
        if p.is_file():
            return str(p)
        raise FileNotFoundError(
            f"Checkpoint not found: {p}\n"
            "Supply --ckpt PATH or set POINTS2SBL_CKPT."
        )

    checked = _default_checkpoint_candidates()
    for p in checked:
        if p.is_file():
            return str(p)

    lines = "\n".join(f"  - {p}" for p in checked)
    raise FileNotFoundError(
        "No default Point Transformer checkpoint was found.\n"
        "For a cloned/editable repository, place the released checkpoint at:\n"
        "  runs\\point_transformer_curated_20260327_170108\\best.pt\n"
        "For a pip/conda installation, run:\n"
        "  points2sbl model download\n"
        "or pass --ckpt PATH / set POINTS2SBL_CKPT.\n"
        "Locations checked:\n"
        f"{lines}"
    )


def _argv_has_dest(argv: Optional[List[str]], dest: str) -> bool:
    """Return True when the user explicitly supplied an option for dest."""
    if argv is None:
        argv = sys.argv[1:]

    opts = set(str(x) for x in argv if str(x).startswith("--"))
    stem_u = dest
    stem_h = dest.replace("_", "-")

    candidates = {
        f"--{stem_u}",
        f"--{stem_h}",
        f"--no-{stem_u}",
        f"--no-{stem_h}",
    }
    return any(c in opts for c in candidates)


def _auto_single_tree_tile_size(xyz: np.ndarray) -> Tuple[float, float]:
    """Return (tile_size_m, max_xy_extent_m) for an isolated-tree cloud.

    The thresholds are intentionally conservative and were chosen from
    points2SBL single-tree validation so that small/medium crowns retain
    local context while large crowns are not fragmented excessively.
    """
    pts = np.asarray(xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return 1.5, 0.0

    xy = pts[:, :2]
    lo = np.nanmin(xy, axis=0)
    hi = np.nanmax(xy, axis=0)
    extent = float(max(0.0, hi[0] - lo[0], hi[1] - lo[1]))

    if extent <= 8.0:
        tile = 1.5
    elif extent <= 15.0:
        tile = 2.5
    elif extent <= 25.0:
        tile = 3.5
    else:
        tile = 5.0

    return float(tile), extent


def _resolve_input_type(
    requested: str,
    xyz: np.ndarray,
    classification: Optional[np.ndarray],
    ground_class_code: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve public input semantics for plot/single_tree/auto.

    auto is conservative:
      * a usable ground-class population => plot
      * otherwise a compact (<=30 m XY extent) cloud => single_tree
      * otherwise => plot (no usable class-2 points will simply be predicted
        as non-ground by the established plot pathway)
    """
    req = str(requested).strip().lower()
    if req not in {"auto", "plot", "single_tree"}:
        raise ValueError(
            f"Unknown input_type={requested!r}. Use auto, plot, or single_tree."
        )

    pts = np.asarray(xyz, dtype=np.float32)
    n = int(pts.shape[0])
    _, extent = _auto_single_tree_tile_size(pts)

    ground_count = 0
    ground_fraction = 0.0
    if classification is not None and len(classification) == n:
        cls = np.asarray(classification)
        ground_count = int(np.count_nonzero(cls == int(ground_class_code)))
        ground_fraction = float(ground_count / max(1, n))

    usable_ground = bool(
        ground_count >= 64
        and ground_fraction >= 0.001
    )

    if req == "plot":
        resolved = "plot"
    elif req == "single_tree":
        resolved = "single_tree"
    else:
        if usable_ground:
            resolved = "plot"
        elif extent <= 30.0:
            resolved = "single_tree"
        else:
            resolved = "plot"

    info = {
        "requested": req,
        "resolved": resolved,
        "xy_extent_m": float(extent),
        "ground_points": int(ground_count),
        "ground_fraction": float(ground_fraction),
        "usable_ground": bool(usable_ground),
    }
    return resolved, info


def _apply_mode_profile(
    args: argparse.Namespace,
    argv: Optional[List[str]],
    runtime_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply backend inference defaults from predict.profiles.<mode>.
    Explicit CLI values always win.
    """
    predict_cfg = runtime_cfg.get("predict", {})
    if not isinstance(predict_cfg, dict):
        return {}

    profiles = predict_cfg.get("profiles", {})
    if not isinstance(profiles, dict):
        return {}

    profile = profiles.get(str(args.mode), {})
    if not isinstance(profile, dict):
        return {}

    flattened: Dict[str, Any] = {}
    for key, value in profile.items():
        if key == "adaptive" and isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened[f"adaptive_{subkey}"] = subvalue
        else:
            flattened[key] = value

    applied: Dict[str, Any] = {}
    for dest, value in flattened.items():
        if not hasattr(args, dest):
            continue
        if _argv_has_dest(argv, dest):
            continue
        setattr(args, dest, value)
        applied[dest] = value

    return applied


def _print_profile_summary(
    mode: str,
    config_path: str,
    checkpoint_path: str,
    applied: Dict[str, Any],
    verbose: bool = False,
) -> None:
    """Print detailed backend profile information only in verbose mode."""
    if not verbose:
        return

    print(f"[INFO] Mode profile: {mode}")
    print(f"[INFO] Runtime config: {config_path}")
    print(f"[INFO] Resolved checkpoint: {checkpoint_path}")

    if applied:
        parts = ", ".join(
            f"{key}={value}"
            for key, value in sorted(applied.items())
        )
        print(f"[INFO] Backend profile defaults: {parts}")
    else:
        print("[INFO] Backend profile defaults: none")



def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_prefix_if_present(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    keys = list(sd.keys())
    if all(k.startswith(prefix) for k in keys):
        n = len(prefix)
        return {k[n:]: v for k, v in sd.items()}
    return sd


def _load_ckpt(path: str) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    ckpt = _torch_load(path)

    cfg = None
    if isinstance(ckpt, dict) and "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg = ckpt["config"]

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        sd = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        sd = ckpt
    else:
        raise RuntimeError(f"Unrecognized checkpoint format at: {path} (type={type(ckpt)})")

    sd = _strip_prefix_if_present(sd, "module.")
    return sd, cfg


def _enable_tf32(
    enable: bool,
    verbose: bool = False,
):
    if not enable:
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if verbose:
        print("[INFO] TF32 enabled.")


def _autocast(enabled: bool):
    if not enabled:
        return torch.no_grad()
    try:
        return torch.amp.autocast("cuda")
    except Exception:
        return torch.cuda.amp.autocast()


def _safe_ckdtree():
    try:
        from scipy.spatial import cKDTree
        return cKDTree
    except Exception as e:
        raise ImportError("scipy is required for predict.py (cKDTree). Install scipy in this env.") from e


GEOM_NAMES = [
    "nx", "ny", "nz",
    "linearity", "planarity", "scattering", "omnivariance",
    "anisotropy", "eigenentropy", "curvature",
]


def normals_and_eigenfeatures(xyz: np.ndarray, k: int = 24) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    n = int(xyz.shape[0])
    if n == 0:
        return np.zeros((0, 10), dtype=np.float32)
    if n < 3:
        return np.zeros((n, 10), dtype=np.float32)

    k = int(max(3, min(int(k), n)))
    cKDTree = _safe_ckdtree()
    tree = cKDTree(xyz)

    _, idx = tree.query(xyz, k=k, workers=1)
    if idx.ndim == 1:
        idx = idx[:, None]
    neigh = xyz[idx]

    mu = neigh.mean(axis=1, keepdims=True)
    X = neigh - mu

    cov = np.einsum("nki,nkj->nij", X, X) / max(1.0, float(k - 1))
    evals, evecs = np.linalg.eigh(cov)

    l0 = np.maximum(evals[:, 2], 1e-12)
    l1 = np.maximum(evals[:, 1], 1e-12)
    l2 = np.maximum(evals[:, 0], 1e-12)

    normals = evecs[:, :, 0].astype(np.float32)

    linearity = (l0 - l1) / l0
    planarity = (l1 - l2) / l0
    scattering = l2 / l0
    omnivariance = np.cbrt(np.maximum(l0 * l1 * l2, 1e-36))
    anisotropy = (l0 - l2) / l0

    s = np.maximum(l0 + l1 + l2, 1e-12)
    p0 = l0 / s
    p1 = l1 / s
    p2 = l2 / s

    eigenentropy = -(p0 * np.log(np.maximum(p0, 1e-12)) +
                     p1 * np.log(np.maximum(p1, 1e-12)) +
                     p2 * np.log(np.maximum(p2, 1e-12)))
    curvature = l2 / s

    eigen_feats = np.stack(
        [linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy, curvature],
        axis=1,
    ).astype(np.float32)

    return np.concatenate([normals, eigen_feats], axis=1).astype(np.float32)


def _geom_select(geom10: np.ndarray, select: List[str], log_omnivariance: bool) -> np.ndarray:
    name_to_col = {n: i for i, n in enumerate(GEOM_NAMES)}
    cols = []
    for s in select:
        if s not in name_to_col:
            raise ValueError(f"Unknown geom_select feature '{s}'. Valid: {GEOM_NAMES}")
        cols.append(name_to_col[s])

    out = geom10[:, cols].astype(np.float32)

    if log_omnivariance and "omnivariance" in select:
        j = select.index("omnivariance")
        out[:, j] = np.log1p(np.maximum(out[:, j], 0.0)).astype(np.float32)

    return out


def _infer_geom_k(feat_cfg: Dict[str, Any]) -> int:
    if not isinstance(feat_cfg, dict):
        return 24

    if "geom_k_infer" in feat_cfg and feat_cfg["geom_k_infer"] is not None:
        try:
            return int(feat_cfg["geom_k_infer"])
        except Exception:
            pass

    gk = feat_cfg.get("geom_k", 24)
    if isinstance(gk, (list, tuple)) and len(gk) > 0:
        return int(gk[0])
    return int(gk)


def build_features_for_block(
    pts_centered: np.ndarray,
    pts_global: np.ndarray,
    feat_cfg: Dict[str, Any],
    expect_dim: int,
    geom_cached_selected: Optional[np.ndarray] = None,
) -> np.ndarray:
    f = feat_cfg or {}

    use_xyz = bool(f.get("use_xyz", True))
    use_centered_xyz = bool(f.get("use_centered_xyz", False))

    include_geom = bool(f.get("include_geom_features", False))
    geom_k = _infer_geom_k(f)
    geom_select = f.get("geom_select", None)
    if geom_select is None:
        geom_select = []
    geom_select = list(geom_select) if isinstance(geom_select, (list, tuple)) else []

    normalize_features = bool(f.get("normalize_features", True))
    standardize_features = bool(f.get("standardize_features", True))
    clipv = float(f.get("clip_features", 5.0))
    log_omnivariance = bool(f.get("log_omnivariance", False))

    pos = np.asarray(pts_global, dtype=np.float32)
    cen = np.asarray(pts_centered, dtype=np.float32)

    geom_block = None
    if include_geom:
        if geom_cached_selected is not None:
            geom_block = np.asarray(geom_cached_selected, dtype=np.float32)
        else:
            geom10 = normals_and_eigenfeatures(pos, k=int(geom_k))
            if len(geom_select) == 0:
                geom_block = geom10.astype(np.float32)
            else:
                geom_block = _geom_select(geom10, geom_select, log_omnivariance).astype(np.float32)

    def cat(parts: List[np.ndarray], has_pos: bool, has_cen: bool):
        if not parts:
            return None
        X = np.concatenate(parts, axis=1).astype(np.float32)
        return (X, has_pos, has_cen)

    candidates: List[Tuple[np.ndarray, bool, bool]] = []

    partsA: List[np.ndarray] = []
    has_posA = False
    has_cenA = False
    if use_xyz:
        partsA.append(pos); has_posA = True
    if use_centered_xyz:
        partsA.append(cen); has_cenA = True
    if include_geom and geom_block is not None:
        partsA.append(geom_block)
    outA = cat(partsA, has_posA, has_cenA)
    if outA is not None:
        candidates.append(outA)

    if use_xyz and use_centered_xyz:
        partsB: List[np.ndarray] = [cen]
        if include_geom and geom_block is not None:
            partsB.append(geom_block)
        candidates.append((np.concatenate(partsB, axis=1).astype(np.float32), False, True))

    if use_xyz and use_centered_xyz:
        partsC: List[np.ndarray] = [pos]
        if include_geom and geom_block is not None:
            partsC.append(geom_block)
        candidates.append((np.concatenate(partsC, axis=1).astype(np.float32), True, False))

    if use_centered_xyz:
        candidates.append((cen.astype(np.float32), False, True))

    if use_xyz:
        candidates.append((pos.astype(np.float32), True, False))

    chosen: Optional[Tuple[np.ndarray, bool, bool]] = None
    for X, hp, hc in candidates:
        if X.shape[1] == int(expect_dim):
            chosen = (X, hp, hc)
            break

    if chosen is None:
        got = candidates[0][0].shape[1] if candidates else -1
        raise RuntimeError(
            f"Feature dim mismatch in predict: got {got} expected {expect_dim}. "
            f"YAML(use_xyz={use_xyz}, use_centered_xyz={use_centered_xyz}, include_geom={include_geom}, "
            f"geom_select={geom_select})."
        )

    x, has_pos, has_cen = chosen
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if normalize_features:
        r = np.linalg.norm(cen, axis=1)
        scale = float(np.median(r) + 1e-6)

        col = 0
        if has_pos and x.shape[1] >= col + 3:
            x[:, col:col + 3] /= scale
            col += 3
        if has_cen and x.shape[1] >= col + 3:
            x[:, col:col + 3] /= scale
            col += 3

    if standardize_features:
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True) + 1e-6
        x = (x - mu) / sd

    if clipv > 0:
        x = np.clip(x, -clipv, clipv).astype(np.float32)

    return x


def _attribute_available(cloud, name: str) -> bool:
    try:
        arr = getattr(cloud, name)
        return arr is not None and len(arr) > 0
    except Exception:
        return False


def _strict_noise_mask_xyz(xyz: np.ndarray, k: int = 12, zscore_thr: float = 4.0) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    n = int(xyz.shape[0])
    if n < max(10, k + 1):
        return np.zeros(n, dtype=bool)

    cKDTree = _safe_ckdtree()
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=min(k + 1, n), workers=1)
    mean_knn = dists[:, 1:].mean(axis=1)
    mu = float(mean_knn.mean())
    sd = float(mean_knn.std()) + 1e-6
    z = (mean_knn - mu) / sd
    return z > float(zscore_thr)


def _strict_noise_mask_radiometric(
    xyz: np.ndarray,
    attr: np.ndarray,
    k: int = 40,
    nsigma: float = 0.80,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    attr = np.asarray(attr, dtype=np.float32)
    n = int(xyz.shape[0])
    if n < max(20, k + 1):
        return np.zeros(n, dtype=bool)

    cKDTree = _safe_ckdtree()
    tree = cKDTree(xyz)
    dists, nn = tree.query(xyz, k=min(k + 1, n), workers=1)

    d = dists[:, 1:]
    nn = nn[:, 1:]

    local_attr = attr[nn]
    local_mu = local_attr.mean(axis=1)
    local_sd = local_attr.std(axis=1) + 1e-6

    z_attr = (attr - local_mu) / local_sd

    mean_knn = d.mean(axis=1)
    mu_d = float(mean_knn.mean())
    sd_d = float(mean_knn.std()) + 1e-6
    z_dist = (mean_knn - mu_d) / sd_d

    # strict outliers: both geometrically sparse and radiometrically weak
    mask = (z_dist > float(nsigma)) & (z_attr < -float(nsigma))
    return mask


def _choose_radiometric_attribute(cloud) -> Optional[str]:
    candidates = [
        "intensity",
        "red",
        "green",
        "blue",
        "nir",
    ]
    for name in candidates:
        if _attribute_available(cloud, name):
            try:
                arr = np.asarray(getattr(cloud, name))
                if arr.size > 0 and np.std(arr.astype(np.float32)) > 1e-6:
                    return name
            except Exception:
                pass
    return None


def _tile_group_indices(xy: np.ndarray, sx: float, sy: float, offset: Tuple[float, float]) -> List[np.ndarray]:
    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)

    ox, oy = float(offset[0]), float(offset[1])
    xmin = float(x.min()) + ox
    ymin = float(y.min()) + oy

    ix = np.floor((x - xmin) / sx).astype(np.int64)
    iy = np.floor((y - ymin) / sy).astype(np.int64)

    key = (ix << 32) ^ (iy & 0xFFFFFFFF)
    order = np.argsort(key, kind="mergesort")
    key_sorted = key[order]

    _, start = np.unique(key_sorted, return_index=True)
    groups: List[np.ndarray] = []
    for j in range(len(start)):
        a = start[j]
        b = start[j + 1] if j + 1 < len(start) else order.size
        groups.append(order[a:b])
    return groups


def _chunk_no_replacement(idx: np.ndarray, n_points: int, rng: np.random.Generator) -> List[np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return []
    idx = idx.copy()
    rng.shuffle(idx)

    chunks: List[np.ndarray] = []
    n = idx.size
    if n <= n_points:
        pad = rng.integers(0, n, size=(n_points - n), dtype=np.int64)
        chunks.append(np.concatenate([idx, idx[pad]], axis=0))
        return chunks

    n_full = n // n_points
    rem = n % n_points

    for i in range(n_full):
        chunks.append(idx[i * n_points:(i + 1) * n_points])

    if rem > 0:
        last = idx[n_full * n_points:]
        pad = rng.integers(0, last.size, size=(n_points - rem), dtype=np.int64)
        chunks.append(np.concatenate([last, last[pad]], axis=0))

    return chunks


def _tile_quality_checks(
    pts_global: np.ndarray,
    tile_size: float,
    min_points: int,
    occupancy_grid_n: int = 4,
    min_occupancy_frac: float = 0.20,
    max_dominant_cell_frac: float = 0.70,
    min_xy_spread_frac: float = 0.20,
) -> Tuple[bool, Dict[str, float]]:
    n = int(pts_global.shape[0])
    metrics: Dict[str, float] = {
        "n_points": float(n),
        "occupancy_frac": 0.0,
        "dominant_cell_frac": 1.0,
        "xy_spread_frac_x": 0.0,
        "xy_spread_frac_y": 0.0,
        "quality_score": 0.0,
    }

    if n < int(min_points):
        return False, metrics

    xy = np.asarray(pts_global[:, :2], dtype=np.float32)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)

    spread_x = float(max(0.0, xmax - xmin))
    spread_y = float(max(0.0, ymax - ymin))

    metrics["xy_spread_frac_x"] = spread_x / max(tile_size, 1e-6)
    metrics["xy_spread_frac_y"] = spread_y / max(tile_size, 1e-6)

    if metrics["xy_spread_frac_x"] < float(min_xy_spread_frac):
        return False, metrics
    if metrics["xy_spread_frac_y"] < float(min_xy_spread_frac):
        return False, metrics

    g = int(max(2, occupancy_grid_n))
    bx = np.floor((xy[:, 0] - xmin) / max(spread_x + 1e-6, 1e-6) * g).astype(np.int64)
    by = np.floor((xy[:, 1] - ymin) / max(spread_y + 1e-6, 1e-6) * g).astype(np.int64)
    bx = np.clip(bx, 0, g - 1)
    by = np.clip(by, 0, g - 1)
    cell_id = bx * g + by
    _, counts = np.unique(cell_id, return_counts=True)

    occupancy_frac = float(len(counts) / float(g * g))
    dominant_cell_frac = float(counts.max() / max(1, n))

    metrics["occupancy_frac"] = occupancy_frac
    metrics["dominant_cell_frac"] = dominant_cell_frac

    if occupancy_frac < float(min_occupancy_frac):
        return False, metrics
    if dominant_cell_frac > float(max_dominant_cell_frac):
        return False, metrics

    q_occ = np.clip(occupancy_frac / max(min_occupancy_frac, 1e-6), 0.0, 1.0)
    q_dom = np.clip(1.0 - dominant_cell_frac / max(max_dominant_cell_frac, 1e-6), 0.0, 1.0)
    q_spx = np.clip(metrics["xy_spread_frac_x"], 0.0, 1.0)
    q_spy = np.clip(metrics["xy_spread_frac_y"], 0.0, 1.0)
    metrics["quality_score"] = float(np.clip(0.30 * q_occ + 0.30 * q_dom + 0.20 * q_spx + 0.20 * q_spy, 0.0, 1.0))

    return True, metrics


def _make_offsets(votes: int, sx: float, sy: float, mode: str, rng: np.random.Generator) -> List[Tuple[float, float]]:
    votes = int(max(1, votes))
    mode = str(mode).lower()

    if mode == "grid4":
        base = [(0.0, 0.0)]
        if votes >= 2:
            base.append((0.5 * sx, 0.5 * sy))
        if votes >= 4:
            base = [(0.0, 0.0), (0.5 * sx, 0.0), (0.0, 0.5 * sy), (0.5 * sx, 0.5 * sy)]
        if votes > len(base):
            extra = [(float(rng.uniform(0.0, sx)), float(rng.uniform(0.0, sy))) for _ in range(votes - len(base))]
            return base + extra
        return base[:votes]

    if mode == "grid8":
        # Deterministic 8-layout stratified offsets.
        #
        # The first four layouts retain the familiar half-tile grid
        # positions. The next four shift the grid to quarter/three-quarter
        # positions so points are observed under substantially different
        # tile-boundary contexts.
        #
        # Existing grid4/random behaviour is intentionally unchanged.
        base = [
            (0.00 * sx, 0.00 * sy),
            (0.50 * sx, 0.00 * sy),
            (0.00 * sx, 0.50 * sy),
            (0.50 * sx, 0.50 * sy),
            (0.25 * sx, 0.25 * sy),
            (0.75 * sx, 0.25 * sy),
            (0.25 * sx, 0.75 * sy),
            (0.75 * sx, 0.75 * sy),
        ]
        if votes > len(base):
            extra = [
                (float(rng.uniform(0.0, sx)), float(rng.uniform(0.0, sy)))
                for _ in range(votes - len(base))
            ]
            return base + extra
        return base[:votes]

    if mode == "hybrid8":
        fixed = [
            (0.00 * sx, 0.00 * sy),
            (0.50 * sx, 0.00 * sy),
            (0.00 * sx, 0.50 * sy),
            (0.50 * sx, 0.50 * sy),
        ]
        if votes <= 4:
            return fixed[:votes]
        extra_n = votes - 4
        extra = [
            (float(rng.uniform(0.0, sx)), float(rng.uniform(0.0, sy)))
            for _ in range(extra_n)
        ]
        return fixed + extra

    if mode == "random":
        return [(float(rng.uniform(0.0, sx)), float(rng.uniform(0.0, sy))) for _ in range(votes)]

    raise ValueError(
        f"Unknown vote_mode={mode}. Use grid4, grid8, hybrid8, or random."
    )



def _analyze_leaf_probability_distribution(
    p_leaf: np.ndarray,
    bins: int = 256,
    smooth_sigma: float = 2.0,
    shoulder_fraction: float = 0.20,
    min_transition_width: float = 0.10,
) -> Dict[str, Any]:
    """
    Analyze the empirical leaf-probability distribution on PREDICTION
    POINTS ONLY. Ground/preserved points are not included because those
    are filled with ground_prob_fill only when the full LAS is written.

    Returns data-driven wood/leaf peaks, the inter-peak valley, and two
    transition-zone shoulders. No Gaussian assumption is required.
    """
    p = np.asarray(p_leaf, dtype=np.float32).reshape(-1)
    p = p[np.isfinite(p)]
    p = p[(p >= 0.0) & (p <= 1.0)]

    if p.size < 32:
        raise RuntimeError(
            "Not enough valid [0,1] prediction probabilities for adaptive analysis."
        )

    bins = int(max(64, min(int(bins), 1024)))
    hist, edges = np.histogram(p, bins=bins, range=(0.0, 1.0), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = hist.astype(np.float64)

    try:
        from scipy.ndimage import gaussian_filter1d
        smooth = gaussian_filter1d(
            hist,
            sigma=float(max(0.5, smooth_sigma)),
            mode="nearest",
        )
    except Exception:
        # Lightweight fallback if scipy.ndimage is unavailable.
        radius = max(1, int(round(2.0 * float(max(0.5, smooth_sigma)))))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        k = np.exp(-(x * x) / (2.0 * float(max(0.5, smooth_sigma)) ** 2))
        k /= max(k.sum(), 1e-12)
        smooth = np.convolve(hist, k, mode="same")

    mid_i = int(np.searchsorted(centers, 0.5))
    mid_i = int(np.clip(mid_i, 2, centers.size - 2))

    left_peak_i = int(np.argmax(smooth[:mid_i]))
    right_peak_i = int(mid_i + np.argmax(smooth[mid_i:]))

    # Guard against a degenerate ordering.
    if right_peak_i <= left_peak_i + 2:
        left_peak_i = int(np.argmax(smooth[: max(2, centers.size // 3)]))
        start = max(left_peak_i + 3, 2 * centers.size // 3)
        right_peak_i = int(start + np.argmax(smooth[start:]))

    valley_slice = smooth[left_peak_i:right_peak_i + 1]
    valley_i = int(left_peak_i + np.argmin(valley_slice))

    left_peak_density = float(smooth[left_peak_i])
    right_peak_density = float(smooth[right_peak_i])
    valley_density = float(smooth[valley_i])

    shoulder_fraction = float(np.clip(shoulder_fraction, 0.005, 0.45))

    left_target = valley_density + shoulder_fraction * max(
        0.0, left_peak_density - valley_density
    )
    right_target = valley_density + shoulder_fraction * max(
        0.0, right_peak_density - valley_density
    )

    # Last left-side bin before the valley whose density is still above
    # the chosen shoulder level.
    left_candidates = np.where(
        smooth[left_peak_i:valley_i + 1] >= left_target
    )[0]
    if left_candidates.size:
        transition_low_i = int(left_peak_i + left_candidates[-1])
    else:
        transition_low_i = int(max(left_peak_i, valley_i - 1))

    # First right-side bin after the valley whose density rises above
    # the right shoulder level.
    right_candidates = np.where(
        smooth[valley_i:right_peak_i + 1] >= right_target
    )[0]
    if right_candidates.size:
        transition_high_i = int(valley_i + right_candidates[0])
    else:
        transition_high_i = int(min(right_peak_i, valley_i + 1))

    wood_peak = float(centers[left_peak_i])
    leaf_peak = float(centers[right_peak_i])
    valley = float(centers[valley_i])
    transition_low = float(centers[transition_low_i])
    transition_high = float(centers[transition_high_i])

    # Ensure a meaningful transition interval. If the density shoulders
    # collapse around a sharp valley, expand symmetrically but remain
    # between the two detected modes.
    min_transition_width = float(np.clip(min_transition_width, 0.02, 0.50))
    if transition_high - transition_low < min_transition_width:
        half = 0.5 * min_transition_width
        transition_low = max(wood_peak, valley - half)
        transition_high = min(leaf_peak, valley + half)

    # Conservative bounds: anchors must stay on their respective sides
    # of the inter-peak valley.
    transition_low = float(np.clip(transition_low, wood_peak, valley))
    transition_high = float(np.clip(transition_high, valley, leaf_peak))

    wood_mask = p <= transition_low
    leaf_mask = p >= transition_high
    trans_mask = ~(wood_mask | leaf_mask)

    total = int(p.size)
    return {
        "n_valid": total,
        "bins": int(bins),
        "centers": centers.astype(np.float32),
        "hist": hist.astype(np.float64),
        "smooth": smooth.astype(np.float64),
        "wood_peak": wood_peak,
        "leaf_peak": leaf_peak,
        "valley": valley,
        "transition_low": transition_low,
        "transition_high": transition_high,
        "wood_anchor_frac": float(np.mean(wood_mask)),
        "transition_frac": float(np.mean(trans_mask)),
        "leaf_anchor_frac": float(np.mean(leaf_mask)),
    }


def _write_probability_distribution_artifacts(
    analysis: Dict[str, Any],
    out_las: str,
) -> Tuple[str, str]:
    """
    Write a dependency-free CSV and SVG diagnostic for pred_leaf_prob.
    Ground class values (-1 by default) are intentionally absent because
    analysis is computed from prediction points only.
    """
    stem = os.path.splitext(os.path.abspath(out_las))[0]
    csv_path = stem + "_pred_leaf_prob_distribution.csv"
    svg_path = stem + "_pred_leaf_prob_distribution.svg"

    centers = np.asarray(analysis["centers"], dtype=np.float64)
    hist = np.asarray(analysis["hist"], dtype=np.float64)
    smooth = np.asarray(analysis["smooth"], dtype=np.float64)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("probability_center,count,smoothed_count\n")
        for x, y, ys in zip(centers, hist, smooth):
            f.write(f"{x:.8f},{int(round(y))},{ys:.8f}\n")

    # Simple publication-quality vector SVG with no matplotlib dependency.
    W, H = 1200, 700
    ml, mr, mt, mb = 95, 45, 55, 95
    pw = W - ml - mr
    ph = H - mt - mb
    ymax = float(max(np.max(hist), np.max(smooth), 1.0))

    def sx(x: float) -> float:
        return ml + float(x) * pw

    def sy(y: float) -> float:
        return mt + ph - (float(y) / ymax) * ph

    smooth_pts = " ".join(
        f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(centers, smooth)
    )

    marks = [
        ("Wood peak", float(analysis["wood_peak"]), "#1f4e79"),
        ("Transition low", float(analysis["transition_low"]), "#7f8c8d"),
        ("Valley", float(analysis["valley"]), "#8e44ad"),
        ("Transition high", float(analysis["transition_high"]), "#7f8c8d"),
        ("Leaf peak", float(analysis["leaf_peak"]), "#1b7f3a"),
    ]

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(
        f'<text x="{W/2:.1f}" y="30" text-anchor="middle" '
        'font-family="Arial" font-size="24" font-weight="bold">'
        'Predicted leaf-probability distribution (non-ground prediction points)</text>'
    )

    # Histogram bars.
    bw = pw / max(1, centers.size)
    for x, y in zip(centers, hist):
        x0 = sx(x) - 0.5 * bw
        y0 = sy(y)
        h = mt + ph - y0
        svg.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{max(0.5,bw):.2f}" '
            f'height="{max(0.0,h):.2f}" fill="#d9e2f3" stroke="none"/>'
        )

    # Smoothed density.
    svg.append(
        f'<polyline points="{smooth_pts}" fill="none" stroke="#111111" '
        'stroke-width="3"/>'
    )

    # Axes.
    svg.append(
        f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" '
        'stroke="black" stroke-width="2"/>'
    )
    svg.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" '
        'stroke="black" stroke-width="2"/>'
    )

    for i in range(0, 11):
        x = i / 10.0
        xx = sx(x)
        svg.append(
            f'<line x1="{xx:.2f}" y1="{mt+ph}" x2="{xx:.2f}" y2="{mt+ph+8}" '
            'stroke="black" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{xx:.2f}" y="{mt+ph+30}" text-anchor="middle" '
            f'font-family="Arial" font-size="15">{x:.1f}</text>'
        )

    svg.append(
        f'<text x="{ml+pw/2:.2f}" y="{H-28}" text-anchor="middle" '
        'font-family="Arial" font-size="19">pred_leaf_prob</text>'
    )
    svg.append(
        f'<text transform="translate(28,{mt+ph/2:.2f}) rotate(-90)" '
        'text-anchor="middle" font-family="Arial" font-size="19">Point count</text>'
    )

    # Transition-zone shading.
    xlo = sx(float(analysis["transition_low"]))
    xhi = sx(float(analysis["transition_high"]))
    svg.append(
        f'<rect x="{xlo:.2f}" y="{mt}" width="{max(0.0,xhi-xlo):.2f}" '
        f'height="{ph}" fill="#f5e6a8" opacity="0.28"/>'
    )

    # Markers and labels.
    label_y = mt + 22
    for j, (name, value, color) in enumerate(marks):
        xx = sx(value)
        svg.append(
            f'<line x1="{xx:.2f}" y1="{mt}" x2="{xx:.2f}" y2="{mt+ph}" '
            f'stroke="{color}" stroke-width="2" stroke-dasharray="7,5"/>'
        )
        ytxt = label_y + (j % 2) * 25
        svg.append(
            f'<text x="{xx:.2f}" y="{ytxt:.2f}" text-anchor="middle" '
            f'font-family="Arial" font-size="14" fill="{color}">'
            f'{name}: {value:.3f}</text>'
        )

    # Footer fractions.
    footer = (
        f"Wood anchor: {100.0*float(analysis['wood_anchor_frac']):.2f}%   |   "
        f"Transition: {100.0*float(analysis['transition_frac']):.2f}%   |   "
        f"Leaf anchor: {100.0*float(analysis['leaf_anchor_frac']):.2f}%"
    )
    svg.append(
        f'<text x="{ml+pw/2:.2f}" y="{H-5}" text-anchor="middle" '
        f'font-family="Arial" font-size="14">{footer}</text>'
    )

    svg.append("</svg>")

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))

    return csv_path, svg_path


def _robust_geometry_prototypes(
    geom: np.ndarray,
    wood_anchor: np.ndarray,
    leaf_anchor: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Robust per-scene geometric prototypes from probability anchors.
    Uses median and MAD so isolated label/probability errors have low influence.
    """
    g = np.asarray(geom, dtype=np.float32)
    w = np.asarray(wood_anchor, dtype=bool)
    l = np.asarray(leaf_anchor, dtype=bool)

    if g.ndim != 2 or g.shape[0] != w.size or w.size != l.size:
        raise ValueError("Geometry/prototype dimensions do not match.")
    if int(w.sum()) < 32 or int(l.sum()) < 32:
        raise RuntimeError("Too few wood/leaf anchor points for robust geometry prototypes.")

    def stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        med = np.median(x, axis=0).astype(np.float32)
        mad = np.median(np.abs(x - med[None, :]), axis=0).astype(np.float32)
        scale = np.maximum(1.4826 * mad, 1e-3).astype(np.float32)
        return med, scale

    wm, ws = stats(g[w])
    lm, ls = stats(g[l])
    return {
        "wood_median": wm,
        "wood_scale": ws,
        "leaf_median": lm,
        "leaf_scale": ls,
    }


def _resolve_transition_adaptive(
    p_leaf: np.ndarray,
    geom: np.ndarray,
    analysis: Dict[str, Any],
    local_woody_support: np.ndarray,
    local_leaf_support: np.ndarray,
    vote_var: np.ndarray,
    vote_count: np.ndarray,
    geom_ratio: float = 0.85,
    local_support_min: float = 0.55,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Lock probability anchors and post-process ONLY the transition zone.
    Transitional points first use robust multivariate geometry similarity,
    then local semantic support, then the empirical valley as the fallback.
    """
    p = np.asarray(p_leaf, dtype=np.float32)
    g = np.asarray(geom, dtype=np.float32)
    lw = np.asarray(local_woody_support, dtype=np.float32)
    ll = np.asarray(local_leaf_support, dtype=np.float32)

    lo = float(analysis["transition_low"])
    hi = float(analysis["transition_high"])
    valley = float(analysis["valley"])

    wood_anchor = p <= lo
    leaf_anchor = p >= hi
    transition = ~(wood_anchor | leaf_anchor)

    pred = np.zeros(p.shape[0], dtype=np.uint8)
    pred[leaf_anchor] = 1

    proto = _robust_geometry_prototypes(g, wood_anchor, leaf_anchor)

    # Robust standardized squared distances; averages make the score
    # comparable across the configured number of geometric channels.
    dw = np.mean(
        ((g - proto["wood_median"][None, :]) / proto["wood_scale"][None, :]) ** 2,
        axis=1,
    )
    dl = np.mean(
        ((g - proto["leaf_median"][None, :]) / proto["leaf_scale"][None, :]) ** 2,
        axis=1,
    )

    unresolved = transition.copy()

    geom_wood = transition & (dw <= float(geom_ratio) * dl)
    geom_leaf = transition & (dl <= float(geom_ratio) * dw)

    pred[geom_wood] = 0
    pred[geom_leaf] = 1
    unresolved &= ~(geom_wood | geom_leaf)

    local_wood = (
        unresolved
        & (lw >= float(local_support_min))
        & (lw > ll)
    )
    local_leaf = (
        unresolved
        & (ll >= float(local_support_min))
        & (ll > lw)
    )

    pred[local_wood] = 0
    pred[local_leaf] = 1
    unresolved &= ~(local_wood | local_leaf)

    # Final fallback only for points that remain ambiguous after geometry
    # and neighborhood evidence.
    pred[unresolved] = (p[unresolved] >= valley).astype(np.uint8)

    stats = {
        "mode": "adaptive_distribution_geometry",
        "wood_peak": float(analysis["wood_peak"]),
        "leaf_peak": float(analysis["leaf_peak"]),
        "valley": valley,
        "transition_low": lo,
        "transition_high": hi,
        "locked_wood": int(wood_anchor.sum()),
        "locked_leaf": int(leaf_anchor.sum()),
        "transition_total": int(transition.sum()),
        "transition_geom_wood": int(geom_wood.sum()),
        "transition_geom_leaf": int(geom_leaf.sum()),
        "transition_local_wood": int(local_wood.sum()),
        "transition_local_leaf": int(local_leaf.sum()),
        "transition_valley_fallback": int(unresolved.sum()),
        "final_wood": int((pred == 0).sum()),
        "final_leaf": int((pred == 1).sum()),
        "geom_feature_count": int(g.shape[1]),
    }
    return pred, stats


def _compute_local_woody_support(
    xyz: np.ndarray,
    strong_woody_mask: np.ndarray,
    k: int = 16,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    strong_woody_mask = np.asarray(strong_woody_mask, dtype=bool)

    n = int(xyz.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    k = int(max(2, min(int(k), n)))
    cKDTree = _safe_ckdtree()
    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=k, workers=1)
    if nn.ndim == 1:
        nn = nn[:, None]

    frac = strong_woody_mask[nn].mean(axis=1).astype(np.float32)
    return frac


def _compute_local_leaf_support(
    xyz: np.ndarray,
    strong_leaf_mask: np.ndarray,
    k: int = 16,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    strong_leaf_mask = np.asarray(strong_leaf_mask, dtype=bool)

    n = int(xyz.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    k = int(max(2, min(int(k), n)))
    cKDTree = _safe_ckdtree()
    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=k, workers=1)
    if nn.ndim == 1:
        nn = nn[:, None]

    frac = strong_leaf_mask[nn].mean(axis=1).astype(np.float32)
    return frac


def _knn_smooth_uncertain_only(
    xyz: np.ndarray,
    cls: np.ndarray,
    p_leaf: np.ndarray,
    vote_var: np.ndarray,
    vote_count: np.ndarray,
    k: int = 24,
    tau: float = 0.60,
    margin: float = 0.08,
    passes: int = 1,
    seam_vote_min: int = 2,
    vote_var_thr: float = 0.015,
) -> Tuple[np.ndarray, int]:
    xyz = np.asarray(xyz, dtype=np.float32)
    cls = np.asarray(cls, dtype=np.uint8).copy()
    p_leaf = np.asarray(p_leaf, dtype=np.float32)
    vote_var = np.asarray(vote_var, dtype=np.float32)
    vote_count = np.asarray(vote_count, dtype=np.float32)

    n = int(xyz.shape[0])
    if n == 0:
        return cls, 0

    k = int(max(2, min(int(k), n)))
    tau = float(np.clip(tau, 0.50, 0.99))
    margin = float(max(0.0, margin))
    passes = int(max(1, passes))

    cKDTree = _safe_ckdtree()
    tree = cKDTree(xyz)

    _, nn = tree.query(xyz, k=k, workers=1)
    if nn.ndim == 1:
        nn = nn[:, None]

    total_changes = 0
    for _ in range(passes):
        low_margin = np.abs(p_leaf - 0.5) <= margin
        unstable = vote_var >= float(vote_var_thr)
        seam_like = vote_count <= float(seam_vote_min)

        target = low_margin | unstable | seam_like
        if not np.any(target):
            break

        nn_cls = cls[nn]
        frac_leaf = nn_cls.mean(axis=1)
        maj_is_leaf = frac_leaf >= 0.5
        maj_frac = np.where(maj_is_leaf, frac_leaf, 1.0 - frac_leaf)

        flip = target & (maj_frac >= tau) & (cls != maj_is_leaf.astype(np.uint8))
        nflip = int(flip.sum())
        if nflip == 0:
            break

        cls[flip] = maj_is_leaf[flip].astype(np.uint8)
        total_changes += nflip

    return cls, total_changes


def _post_refine_woody_points(
    xyz: np.ndarray,
    cls: np.ndarray,
    p_leaf: np.ndarray,
    k: int = 16,
    min_woody_support: float = 0.28,
    keep_if_p_leaf_le: float = 0.15,
    flip_only_if_p_leaf_ge: float = 0.20,
) -> Tuple[np.ndarray, int]:
    xyz = np.asarray(xyz, dtype=np.float32)
    cls = np.asarray(cls, dtype=np.uint8).copy()
    p_leaf = np.asarray(p_leaf, dtype=np.float32)

    n = int(xyz.shape[0])
    if n == 0:
        return cls, 0

    k = int(max(3, min(int(k), n - 1 if n > 1 else 1)))
    if k < 2:
        return cls, 0

    cKDTree = _safe_ckdtree()
    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=min(k + 1, n), workers=1)
    if nn.ndim == 1:
        nn = nn[:, None]

    if nn.shape[1] <= 1:
        return cls, 0

    nn = nn[:, 1:]
    woody_frac = (cls[nn] == 0).mean(axis=1).astype(np.float32)

    woody_now = (cls == 0)
    strong_woody_keep = p_leaf <= float(keep_if_p_leaf_le)
    weak_or_mid = p_leaf >= float(flip_only_if_p_leaf_ge)
    low_support = woody_frac < float(min_woody_support)

    flip_mask = woody_now & low_support & weak_or_mid & (~strong_woody_keep)
    nflip = int(flip_mask.sum())
    if nflip > 0:
        cls[flip_mask] = 1

    return cls, nflip


def _make_core_woody_mask(
    pred: np.ndarray,
    p_leaf: np.ndarray,
    geom_woody_support: np.ndarray,
    local_woody_support: np.ndarray,
    vote_var: np.ndarray,
    vote_count: np.ndarray,
    core_p_leaf_max: float = 0.18,
    core_min_local_support: float = 0.55,
    core_min_geom_support: float = 0.52,
    core_max_vote_var: float = 0.030,
    core_min_vote_count: float = 2.0,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.uint8)
    p_leaf = np.asarray(p_leaf, dtype=np.float32)
    geom_woody_support = np.asarray(geom_woody_support, dtype=np.float32)
    local_woody_support = np.asarray(local_woody_support, dtype=np.float32)
    vote_var = np.asarray(vote_var, dtype=np.float32)
    vote_count = np.asarray(vote_count, dtype=np.float32)

    core = (
        (pred == 0) &
        (p_leaf <= float(core_p_leaf_max)) &
        (local_woody_support >= float(core_min_local_support)) &
        (geom_woody_support >= float(core_min_geom_support)) &
        (vote_var <= float(core_max_vote_var)) &
        (vote_count >= float(core_min_vote_count))
    )
    return core.astype(bool)


def _post_refine_woody_structure(
    xyz: np.ndarray,
    cls: np.ndarray,
    p_leaf: np.ndarray,
    core_woody_mask: np.ndarray,
    geom_woody_support: np.ndarray,
    local_woody_support: np.ndarray,
    k: int = 16,
    min_core_neighbor_frac: float = 0.20,
    weak_p_leaf_ge: float = 0.24,
    keep_if_geom_ge: float = 0.68,
    keep_if_local_ge: float = 0.72,
) -> Tuple[np.ndarray, int, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float32)
    cls = np.asarray(cls, dtype=np.uint8).copy()
    p_leaf = np.asarray(p_leaf, dtype=np.float32)
    core_woody_mask = np.asarray(core_woody_mask, dtype=bool)
    geom_woody_support = np.asarray(geom_woody_support, dtype=np.float32)
    local_woody_support = np.asarray(local_woody_support, dtype=np.float32)

    n = int(xyz.shape[0])
    if n == 0:
        return cls, 0, np.zeros((0,), dtype=bool)

    k = int(max(3, min(int(k), n - 1 if n > 1 else 1)))
    if k < 2:
        return cls, 0, np.zeros((n,), dtype=bool)

    cKDTree = _safe_ckdtree()
    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=min(k + 1, n), workers=1)
    if nn.ndim == 1:
        nn = nn[:, None]
    if nn.shape[1] <= 1:
        return cls, 0, np.zeros((n,), dtype=bool)

    nn = nn[:, 1:]
    core_frac = core_woody_mask[nn].mean(axis=1).astype(np.float32)

    woody_now = (cls == 0)
    non_core_woody = woody_now & (~core_woody_mask)
    weak_woody = p_leaf >= float(weak_p_leaf_ge)
    weak_structure = core_frac < float(min_core_neighbor_frac)
    strong_keep = (
        (geom_woody_support >= float(keep_if_geom_ge)) |
        (local_woody_support >= float(keep_if_local_ge))
    )

    flip_mask = non_core_woody & weak_woody & weak_structure & (~strong_keep)
    nflip = int(flip_mask.sum())
    if nflip > 0:
        cls[flip_mask] = 1

    return cls, nflip, flip_mask.astype(bool)



def _cleanup_small_woody_components(
    xyz: np.ndarray,
    cls: np.ndarray,
    p_leaf: np.ndarray,
    core_woody_mask: np.ndarray,
    radius: float = 0.12,
    min_component_size: int = 24,
    keep_component_if_core_frac_ge: float = 0.20,
    keep_component_if_mean_p_leaf_le: float = 0.18,
    cleanup_mask: Optional[np.ndarray] = None,
    max_candidates: int = 1000000,
    progress_every: int = 250000,
) -> Tuple[np.ndarray, int, int, np.ndarray]:
    """
    Memory-safe cleanup of small woody fragments.

    Key safeguards:
    - Operate only on cleanup candidates, not all woody points.
    - Skip component cleanup entirely when the candidate set is too large.
    - Query radius neighbors on demand inside BFS instead of materializing the
      full radius graph for every candidate point at once.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    cls = np.asarray(cls, dtype=np.uint8).copy()
    p_leaf = np.asarray(p_leaf, dtype=np.float32)
    core_woody_mask = np.asarray(core_woody_mask, dtype=bool)

    if cleanup_mask is None:
        cleanup_mask = (cls == 0) & (~core_woody_mask)
    else:
        cleanup_mask = np.asarray(cleanup_mask, dtype=bool) & (cls == 0) & (~core_woody_mask)

    woody_idx = np.where(cleanup_mask)[0]
    nw = int(woody_idx.size)
    flip_global = np.zeros((cls.shape[0],), dtype=bool)

    if nw == 0:
        return cls, 0, 0, flip_global

    if nw > int(max_candidates):
        print(
            f"[WARN] Skipping woody component cleanup: "
            f"candidates={nw} exceeds max_candidates={int(max_candidates)}"
        )
        return cls, 0, 0, flip_global

    pts = xyz[woody_idx]
    cKDTree = _safe_ckdtree()
    tree = cKDTree(pts)

    visited = np.zeros((nw,), dtype=bool)
    small_components = 0
    flips = 0

    for start in range(nw):
        if visited[start]:
            continue

        if progress_every > 0 and start > 0 and (start % int(progress_every) == 0):
            seen = int(visited.sum())
            print(f"[INFO] Woody cleanup progress: seeds={start}/{nw} visited={seen}/{nw}")

        stack = [start]
        visited[start] = True
        comp = []

        while stack:
            cur = stack.pop()
            comp.append(cur)
            neigh_cur = tree.query_ball_point(pts[cur], r=float(radius), workers=1)
            for nb in neigh_cur:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        comp = np.asarray(comp, dtype=np.int64)
        comp_global = woody_idx[comp]
        comp_size = int(comp_global.size)

        if comp_size >= int(min_component_size):
            continue

        small_components += 1
        comp_core_frac = float(core_woody_mask[comp_global].mean()) if comp_size > 0 else 0.0
        comp_mean_p_leaf = float(p_leaf[comp_global].mean()) if comp_size > 0 else 1.0

        keep = (
            (comp_core_frac >= float(keep_component_if_core_frac_ge)) |
            (comp_mean_p_leaf <= float(keep_component_if_mean_p_leaf_le))
        )
        if keep:
            continue

        cls[comp_global] = 1
        flip_global[comp_global] = True
        flips += comp_size

    return cls, int(flips), int(small_components), flip_global


def _reassign_uncertain_points(
    xyz: np.ndarray,
    cls: np.ndarray,
    uncertain_mask: np.ndarray,
    p_leaf: np.ndarray,
    core_woody_mask: np.ndarray,
    strong_leaf_mask: np.ndarray,
    k: int = 8,
    woody_neighbor_frac_thr: float = 0.70,
    leaf_neighbor_frac_thr: float = 0.55,
    woody_p_leaf_max: float = 0.28,
) -> Tuple[np.ndarray, int, int]:
    xyz = np.asarray(xyz, dtype=np.float32)
    cls = np.asarray(cls, dtype=np.uint8).copy()
    uncertain_mask = np.asarray(uncertain_mask, dtype=bool)
    p_leaf = np.asarray(p_leaf, dtype=np.float32)
    core_woody_mask = np.asarray(core_woody_mask, dtype=bool)
    strong_leaf_mask = np.asarray(strong_leaf_mask, dtype=bool)

    n = int(xyz.shape[0])
    if n == 0 or not np.any(uncertain_mask):
        return cls, 0, 0

    stable_mask = core_woody_mask | strong_leaf_mask
    stable_mask &= (~uncertain_mask)
    stable_idx = np.where(stable_mask)[0]
    uncertain_idx = np.where(uncertain_mask)[0]
    if stable_idx.size == 0 or uncertain_idx.size == 0:
        return cls, 0, 0

    k_eff = int(max(1, min(int(k), stable_idx.size)))
    cKDTree = _safe_ckdtree()
    tree = cKDTree(xyz[stable_idx])
    _, nn = tree.query(xyz[uncertain_idx], k=k_eff, workers=1)
    if np.ndim(nn) == 1:
        nn = np.asarray(nn, dtype=np.int64)[:, None]

    neigh_global = stable_idx[np.asarray(nn, dtype=np.int64)]
    neigh_cls = cls[neigh_global]
    woody_frac = (neigh_cls == 0).mean(axis=1).astype(np.float32)
    leaf_frac = (neigh_cls == 1).mean(axis=1).astype(np.float32)

    to_woody = (
        (woody_frac >= float(woody_neighbor_frac_thr)) &
        (p_leaf[uncertain_idx] <= float(woody_p_leaf_max))
    )
    to_leaf = (leaf_frac >= float(leaf_neighbor_frac_thr)) | (~to_woody)

    n_woody = int(np.sum(to_woody))
    n_leaf = int(np.sum(to_leaf))
    if n_woody > 0:
        cls[uncertain_idx[to_woody]] = 0
    if n_leaf > 0:
        cls[uncertain_idx[to_leaf]] = 1
    return cls, n_woody, n_leaf


def _final_decision_three_zone(
    p_leaf: np.ndarray,
    geom_woody_support: np.ndarray,
    local_woody_support: np.ndarray,
    local_leaf_support: np.ndarray,
    vote_var: np.ndarray,
    vote_count: np.ndarray,
    t_low: float = 0.35,
    t_high: float = 0.80,
    geom_rescue_thr: float = 0.58,
    local_woody_thr: float = 0.48,
    local_leaf_thr: float = 0.62,
    vote_var_thr: float = 0.020,
    vote_count_low: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, int]]:
    p_leaf = np.asarray(p_leaf, dtype=np.float32)
    geom_woody_support = np.asarray(geom_woody_support, dtype=np.float32)
    local_woody_support = np.asarray(local_woody_support, dtype=np.float32)
    local_leaf_support = np.asarray(local_leaf_support, dtype=np.float32)
    vote_var = np.asarray(vote_var, dtype=np.float32)
    vote_count = np.asarray(vote_count, dtype=np.float32)

    pred = np.zeros_like(p_leaf, dtype=np.uint8)

    strong_woody = p_leaf <= float(t_low)
    strong_leaf = p_leaf >= float(t_high)
    mid = ~(strong_woody | strong_leaf)

    pred[strong_leaf] = 1
    pred[strong_woody] = 0

    woody_rescue_geom = geom_woody_support >= float(geom_rescue_thr)
    woody_rescue_local = local_woody_support >= float(local_woody_thr)

    unstable = vote_var >= float(vote_var_thr)
    lowvote = vote_count <= float(vote_count_low)
    woody_rescue_unstable = (unstable | lowvote) & (local_woody_support >= max(0.30, float(local_woody_thr) - 0.10))

    leaf_confirm_mid = (
        (local_leaf_support >= float(local_leaf_thr)) &
        (geom_woody_support < max(0.0, float(geom_rescue_thr) - 0.10)) &
        (p_leaf >= 0.58)
    )

    mid_woody = mid & (woody_rescue_geom | woody_rescue_local | woody_rescue_unstable)
    mid_leaf = mid & (~mid_woody) & leaf_confirm_mid
    mid_remaining = mid & (~mid_woody) & (~mid_leaf)

    pred[mid_woody] = 0
    pred[mid_leaf] = 1
    pred[mid_remaining] = (p_leaf[mid_remaining] >= 0.62).astype(np.uint8)

    stats = {
        "strong_woody": int(strong_woody.sum()),
        "strong_leaf": int(strong_leaf.sum()),
        "mid_total": int(mid.sum()),
        "mid_woody": int(mid_woody.sum()),
        "mid_leaf": int(mid_leaf.sum()),
        "mid_remaining": int(mid_remaining.sum()),
    }
    return pred, stats


@torch.inference_mode()
def _predict_cli_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=None,
        help=(
            "Runtime YAML. If omitted, the packaged Point Transformer "
            "configuration is used. Advanced users may override it."
        ),
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help=(
            "Checkpoint path. If omitted, points2SBL searches standard "
            "repository/model-cache locations or POINTS2SBL_CKPT."
        ),
    )
    ap.add_argument("--in_las", required=True)
    ap.add_argument("--out_las", required=True)
    ap.add_argument(
        "--input_type",
        "--input-type",
        dest="input_type",
        choices=["auto", "plot", "single_tree"],
        default="auto",
        help=(
            "Input scene semantics. 'plot' preserves/excludes class-2 ground "
            "using the established plot workflow; 'single_tree' assumes an "
            "isolated tree without ground and disables plot denoising/ground "
            "exclusion by default; 'auto' resolves the mode independently for "
            "each input file."
        ),
    )

    ap.add_argument(
        "--mode",
        choices=["full", "raw", "adaptive"],
        default="full",
        help=(
            "Inference mode. 'full' preserves the established points2SBL "
            "pipeline (default). 'raw' writes the direct binary model "
            "decision using --thr and disables optional semantic refinement. "
            "'adaptive' derives wood/leaf anchors and the transition zone "
            "from the empirical pred_leaf_prob distribution and resolves "
            "only transition points using scene-specific geometry/local support."
        ),
    )

    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--tile_size_m", type=float, default=None)
    ap.add_argument("--n_points", "--block_n_points", dest="n_points", type=int, default=None)

    ap.add_argument("--votes", type=int, default=6)
    ap.add_argument("--vote_mode", type=str, default="grid4", choices=["grid4", "grid8", "hybrid8", "random"])
    ap.add_argument("--vote_weight", type=str, default="confidence", choices=["uniform", "confidence"])

    ap.add_argument("--batch_blocks", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min_points", type=int, default=64)

    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true", default=False)
    ap.add_argument("--tf32", action="store_true", default=True)
    ap.add_argument("--no-tf32", dest="tf32", action="store_false")
    ap.add_argument("--geom_cache", type=str, default="all", choices=["none", "all"])

    ap.add_argument("--geom_vote_beta", type=float, default=0.25)
    ap.add_argument("--geom_vote_min", type=float, default=0.30)
    ap.add_argument("--geom_vote_max", type=float, default=1.00)

    ap.add_argument("--spatial_vote", "--spatial-vote", dest="spatial_vote", action="store_true", default=True)
    ap.add_argument("--no-spatial_vote", "--no-spatial-vote", dest="spatial_vote", action="store_false")
    ap.add_argument("--spatial_sigma_frac", type=float, default=0.35)
    ap.add_argument("--spatial_min", type=float, default=0.20)

    ap.add_argument("--smooth", action="store_true", default=True)
    ap.add_argument("--no-smooth", dest="smooth", action="store_false")
    ap.add_argument("--smooth_k", type=int, default=24)
    ap.add_argument("--smooth_tau", type=float, default=0.62)
    ap.add_argument("--smooth_margin", type=float, default=0.08)
    ap.add_argument("--smooth_passes", type=int, default=1)

    ap.add_argument("--adaptive_tile_checks", "--adaptive-tile-checks", dest="adaptive_tile_checks", action="store_true", default=True)
    ap.add_argument("--no-adaptive_tile_checks", "--no-adaptive-tile-checks", dest="adaptive_tile_checks", action="store_false")
    ap.add_argument("--occupancy_grid_n", type=int, default=4)
    ap.add_argument("--min_occupancy_frac", type=float, default=0.20)
    ap.add_argument("--max_dominant_cell_frac", type=float, default=0.70)
    ap.add_argument("--min_xy_spread_frac", type=float, default=0.20)

    ap.add_argument("--t_low", type=float, default=0.35)
    ap.add_argument("--t_high", type=float, default=0.80)
    ap.add_argument("--geom_rescue_thr", type=float, default=0.58)
    ap.add_argument("--local_woody_thr", type=float, default=0.48)
    ap.add_argument("--local_leaf_thr", type=float, default=0.62)
    ap.add_argument("--midband_knn_k", type=int, default=16)
    ap.add_argument("--vote_var_thr", type=float, default=0.020)
    ap.add_argument("--vote_count_low", type=float, default=2.0)

    # Adaptive probability-distribution mode. These parameters affect only
    # --mode adaptive and therefore cannot alter legacy full/raw behavior.
    ap.add_argument("--adaptive_hist_bins", type=int, default=256)
    ap.add_argument("--adaptive_hist_smooth_sigma", type=float, default=2.0)
    ap.add_argument("--adaptive_shoulder_fraction", type=float, default=0.20)
    ap.add_argument("--adaptive_min_transition_width", type=float, default=0.10)
    ap.add_argument("--adaptive_geom_ratio", type=float, default=0.85)
    ap.add_argument("--adaptive_local_support_min", type=float, default=0.55)
    ap.add_argument(
        "--adaptive_write_distribution",
        action="store_true",
        default=True,
    )
    ap.add_argument(
        "--no-adaptive_write_distribution",
        "--no-adaptive-write-distribution",
        dest="adaptive_write_distribution",
        action="store_false",
    )

    ap.add_argument("--exclude_ground_class", "--exclude-ground-class", dest="exclude_ground_class", action="store_true", default=True)
    ap.add_argument("--no-exclude_ground_class", "--no-exclude-ground-class", dest="exclude_ground_class", action="store_false")
    ap.add_argument("--ground_class_code", type=int, default=2)
    ap.add_argument("--ground_prob_fill", type=float, default=-1.0)

    ap.add_argument("--denoise", action="store_true", default=True)
    ap.add_argument("--no-denoise", dest="denoise", action="store_false")
    ap.add_argument("--denoise_preserve_ground", "--denoise-preserve-ground", dest="denoise_preserve_ground", action="store_true", default=True)
    ap.add_argument("--no-denoise_preserve_ground", "--no-denoise-preserve-ground", dest="denoise_preserve_ground", action="store_false")
    ap.add_argument("--rad_sor_k", type=int, default=40)
    ap.add_argument("--rad_sor_nsigma", type=float, default=0.80)
    ap.add_argument("--xyz_sor_k", type=int, default=3)
    ap.add_argument("--xyz_sor_nsigma", type=float, default=1.0)

    ap.add_argument("--post_refine_woody", "--post-refine-woody", dest="post_refine_woody", action="store_true", default=True)
    ap.add_argument("--no-post_refine_woody", "--no-post-refine-woody", dest="post_refine_woody", action="store_false")
    ap.add_argument("--woody_refine_k", type=int, default=16)
    ap.add_argument("--woody_refine_min_support", type=float, default=0.28)
    ap.add_argument("--woody_refine_keep_if_p_leaf_le", type=float, default=0.15)
    ap.add_argument("--woody_refine_flip_only_if_p_leaf_ge", type=float, default=0.20)

    ap.add_argument("--post_refine_woody_structure", "--post-refine-woody-structure", dest="post_refine_woody_structure", action="store_true", default=True)
    ap.add_argument("--no-post_refine_woody_structure", "--no-post-refine-woody-structure", dest="post_refine_woody_structure", action="store_false")
    ap.add_argument("--woody_structure_k", type=int, default=16)
    ap.add_argument("--woody_core_p_leaf_max", type=float, default=0.18)
    ap.add_argument("--woody_core_min_local_support", type=float, default=0.55)
    ap.add_argument("--woody_core_min_geom_support", type=float, default=0.52)
    ap.add_argument("--woody_core_max_vote_var", type=float, default=0.030)
    ap.add_argument("--woody_core_min_vote_count", type=float, default=2.0)
    ap.add_argument("--woody_structure_min_core_neighbor_frac", type=float, default=0.20)
    ap.add_argument("--woody_structure_weak_p_leaf_ge", type=float, default=0.24)
    ap.add_argument("--woody_structure_keep_if_geom_ge", type=float, default=0.68)
    ap.add_argument("--woody_structure_keep_if_local_ge", type=float, default=0.72)

    ap.add_argument("--cleanup_small_woody_components", "--cleanup-small-woody-components", dest="cleanup_small_woody_components", action="store_true", default=True)
    ap.add_argument("--no-cleanup_small_woody_components", "--no-cleanup-small-woody-components", dest="cleanup_small_woody_components", action="store_false")
    ap.add_argument("--woody_component_radius", type=float, default=0.12)
    ap.add_argument("--woody_component_min_size", type=int, default=24)
    ap.add_argument("--woody_component_keep_if_core_frac_ge", type=float, default=0.20)
    ap.add_argument("--woody_component_keep_if_mean_p_leaf_le", type=float, default=0.18)
    ap.add_argument("--woody_component_max_candidates", type=int, default=1000000)
    ap.add_argument("--woody_component_progress_every", type=int, default=250000)

    ap.add_argument("--reassign_uncertain", "--reassign-uncertain", dest="reassign_uncertain", action="store_true", default=True)
    ap.add_argument("--no-reassign_uncertain", "--no-reassign-uncertain", dest="reassign_uncertain", action="store_false")
    ap.add_argument("--uncertain_reassign_k", type=int, default=8)
    ap.add_argument("--uncertain_reassign_woody_neighbor_frac", type=float, default=0.70)
    ap.add_argument("--uncertain_reassign_leaf_neighbor_frac", type=float, default=0.55)
    ap.add_argument("--uncertain_reassign_woody_p_leaf_max", type=float, default=0.28)

    ap.add_argument(
        "--progress",
        type=str,
        default="tiles",
        choices=["none", "tiles"],
    )
    ap.add_argument(
        "--progress_every_tiles",
        type=int,
        default=200,
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show detailed backend and inference diagnostics.",
    )

    ap.add_argument("--overwrite_classification", "--overwrite-classification", dest="overwrite_classification", action="store_true", default=True)
    ap.add_argument("--no-overwrite_classification", "--no-overwrite-classification", dest="overwrite_classification", action="store_false")
    ap.add_argument("--save_pred_class_dim", "--save-pred-class-dim", dest="save_pred_class_dim", action="store_true", default=True)
    ap.add_argument("--no-save_pred_class_dim", "--no-save-pred-class-dim", dest="save_pred_class_dim", action="store_false")
    ap.add_argument("--save_pred_prob_dim", "--save-pred-prob-dim", dest="save_pred_prob_dim", action="store_true", default=True)
    ap.add_argument("--no-save_pred_prob_dim", "--no-save-pred-prob-dim", dest="save_pred_prob_dim", action="store_false")

    args = ap.parse_args(argv)

    # Minimal public interface:
    # points2sbl predict --mode MODE --in_las INPUT --out_las OUTPUT
    # Advanced CLI arguments remain available and override backend profiles.
    args.config = str(args.config or _bundled_point_transformer_config())
    args.ckpt = _resolve_checkpoint_path(args.ckpt)

    runtime_cfg = load_yaml(args.config)
    applied_profile = _apply_mode_profile(
        args=args,
        argv=list(argv) if argv is not None else None,
        runtime_cfg=runtime_cfg,
    )

    _print_profile_summary(
        mode=str(args.mode),
        config_path=str(args.config),
        checkpoint_path=str(args.ckpt),
        applied=applied_profile,
        verbose=bool(args.verbose),
    )

    if args.mode == "raw":
        # Direct model-decision mode. Feature construction, tiling/chunking,
        # ground exclusion, checkpoint handling, and LAS I/O remain unchanged.
        # Only optional semantic post-processing/refinement is disabled.
        args.spatial_vote = False
        args.smooth = False
        args.denoise = False
        args.post_refine_woody = False
        args.post_refine_woody_structure = False
        args.cleanup_small_woody_components = False
        args.reassign_uncertain = False
        print(
            "[INFO] Raw mode enabled: direct model decision; "
            "semantic post-processing/refinement disabled."
        )

    if args.mode == "adaptive":
        # Adaptive mode owns the transition-zone resolution. Global semantic
        # refinement is disabled so high-confidence probability anchors are
        # not subsequently altered. Multi-layout voting remains available.
        args.smooth = False
        args.denoise = False
        args.post_refine_woody = False
        args.post_refine_woody_structure = False
        args.cleanup_small_woody_components = False
        args.reassign_uncertain = False
        print(
            "[INFO] Adaptive mode enabled: probability-distribution anchors "
            "+ transition-only geometry/local resolver."
        )

    if args.mode == "adaptive" and args.geom_cache == "none":
        print(
            "[INFO] Adaptive mode requires scene geometry prototypes; "
            "forcing geom_cache=all for this run."
        )
        args.geom_cache = "all"

    cfg_file = runtime_cfg
    state, cfg_ckpt = _load_ckpt(args.ckpt)
    cfg = cfg_ckpt if cfg_ckpt is not None else cfg_file
    if cfg_ckpt is not None and args.verbose:
        print(
            "[INFO] Using config embedded in checkpoint "
            "(ckpt['config']) to avoid feature/model mismatch."
        )

    model_cfg = cfg.get("model", {})
    feat_cfg = cfg.get("features", {})
    train_cfg = cfg.get("train", {})

    model_name = str(model_cfg.get("name", "point_transformer")).lower()
    num_classes = int(model_cfg.get("num_classes", 2))
    in_dim = int(model_cfg.get("in_dim", 3))

    radius = float(_get(cfg, ["data", "block", "radius"], default=0.75))
    default_tile_size = 2.0 * radius
    default_n_points = int(_get(cfg, ["data", "block", "n_points"], default=8192))
    default_votes = int(_get(cfg, ["predict", "votes"], default=6))
    default_batch_blocks = int(_get(cfg, ["predict", "batch_blocks"], default=16))
    default_seed = int(_get(cfg, ["train", "seed"], default=123))
    default_min_points = int(_get(cfg, ["data", "block", "min_points"], default=64))

    tile_size = float(args.tile_size_m if args.tile_size_m is not None else default_tile_size)
    n_points = int(args.n_points if args.n_points is not None else default_n_points)
    votes_req = int(args.votes if args.votes is not None else default_votes)
    batch_blocks = int(args.batch_blocks if args.batch_blocks is not None else default_batch_blocks)
    seed = int(args.seed if args.seed is not None else default_seed)
    min_points = int(args.min_points if args.min_points is not None else default_min_points)

    h = hashlib.sha1(os.path.abspath(args.in_las).encode("utf-8")).digest()
    file_salt = int.from_bytes(h[:4], "little", signed=False)
    rng = np.random.default_rng(int(seed) ^ file_salt)

    if args.device is not None:
        dev_str = args.device
    else:
        dev_str = str(train_cfg.get("device", "cuda")).lower()
        dev_str = "cuda" if "cuda" in dev_str else "cpu"

    _enable_tf32(
        bool(args.tf32),
        verbose=bool(args.verbose),
    )

    cuda_ok = torch.cuda.is_available()

    if dev_str == "cuda" and not cuda_ok:
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            "Install a CUDA-enabled PyTorch build or explicitly use "
            "--device cpu."
        )

    device = torch.device(
        "cuda" if dev_str == "cuda" else "cpu"
    )

    # Only modify the automatic/default batch size.
    # Explicit --batch_blocks always wins.
    batch_was_explicit = _argv_has_dest(
        list(argv) if argv is not None else None,
        "batch_blocks",
    )

    if (
        device.type == "cuda"
        and not batch_was_explicit
    ):
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = (
                float(props.total_memory)
                / (1024.0 ** 3)
            )

            if vram_gb <= 10.0:
                batch_blocks = min(batch_blocks, 4)

            elif vram_gb <= 18.0:
                batch_blocks = min(batch_blocks, 8)

        except Exception:
            pass

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"

    # Read the cloud before printing the banner so input semantics and the
    # automatic single-tree tile size are resolved up front.
    cloud = read_las(args.in_las)
    xyz_all = np.asarray(cloud.xyz, dtype=np.float32)
    n_all = int(xyz_all.shape[0])
    if n_all == 0:
        raise RuntimeError("Input LAS has 0 points.")

    old_cls_all = None
    if cloud.labels is not None and len(cloud.labels) == n_all:
        old_cls_all = np.asarray(cloud.labels, dtype=np.uint8)

    resolved_input_type, input_type_info = _resolve_input_type(
        requested=str(args.input_type),
        xyz=xyz_all,
        classification=old_cls_all,
        ground_class_code=int(args.ground_class_code),
    )

    # High-level input profiles only supply defaults. Explicit expert CLI
    # switches always win.
    denoise_explicit = _argv_has_dest(
        list(argv) if argv is not None else None,
        "denoise",
    )
    ground_explicit = _argv_has_dest(
        list(argv) if argv is not None else None,
        "exclude_ground_class",
    )
    tile_explicit = _argv_has_dest(
        list(argv) if argv is not None else None,
        "tile_size_m",
    )

    if resolved_input_type == "single_tree":
        if not denoise_explicit:
            args.denoise = False
        if not ground_explicit:
            args.exclude_ground_class = False
        if not tile_explicit:
            tile_size, _ = _auto_single_tree_tile_size(xyz_all)
            args.tile_size_m = float(tile_size)

    print("")
    print("points2SBL v0.3.0")
    print(f"Mode      : {str(args.mode).upper()}")
    print(f"Input type: {resolved_input_type.upper().replace('_', ' ')}")
    print(f"Model     : {model_name}")
    print(f"Device    : {device_name}")
    print(f"Input     : {args.in_las}")
    print(f"Output    : {args.out_las}")
    print(
        "Inference : "
        f"tile={tile_size:g}m | "
        f"block={n_points} | "
        f"votes={votes_req} {args.vote_mode} | "
        f"batch={batch_blocks}"
    )
    print("")

    if args.verbose:
        print(f"[INFO] CKPT      : {args.ckpt}")
        print(f"[INFO] Classes   : {num_classes}")
        print(f"[INFO] in_dim    : {in_dim}")
        print(
            "[INFO] InputType : "
            f"requested={input_type_info['requested']} "
            f"resolved={input_type_info['resolved']} "
            f"xy_extent={input_type_info['xy_extent_m']:.3f}m "
            f"ground_points={input_type_info['ground_points']} "
            f"usable_ground={input_type_info['usable_ground']}"
        )
        print(
            f"[INFO] AMP/TF32  : "
            f"amp={bool(args.amp)} "
            f"tf32={bool(args.tf32)}"
        )
        print(f"[INFO] GeomCache : {args.geom_cache}")
        print(
            f"[INFO] DualThr   : "
            f"t_low={args.t_low} "
            f"t_high={args.t_high}"
        )
        print(
            f"[INFO] MidRescue : "
            f"geom={args.geom_rescue_thr} "
            f"local_wood={args.local_woody_thr} "
            f"local_leaf={args.local_leaf_thr}"
        )
        print(
            f"[INFO] SpatialVote: "
            f"{bool(args.spatial_vote)}"
        )
        print(
            f"[INFO] Smooth    : "
            f"{bool(args.smooth)}"
        )
        print(
            f"[INFO] Denoise   : "
            f"{bool(args.denoise)}"
        )
        print(
            f"[INFO] GroundExcl: "
            f"{bool(args.exclude_ground_class)}"
        )
        print(
            f"[INFO] WoodyRef  : "
            f"{bool(args.post_refine_woody)}"
        )
        print(
            f"[INFO] WoodyCore : "
            f"{bool(args.post_refine_woody_structure)}"
        )
        print(
            f"[INFO] WoodyComp : "
            f"{bool(args.cleanup_small_woody_components)}"
        )

    if old_cls_all is not None:
        u0, c0 = np.unique(old_cls_all, return_counts=True)
        print(f"[INFO] Existing input Classification dist: {dict(zip(u0.tolist(), c0.tolist()))}")

    if resolved_input_type == "single_tree":
        if not tile_explicit:
            print(
                "[INFO] Single-tree defaults: "
                f"denoise={bool(args.denoise)} "
                f"exclude_ground={bool(args.exclude_ground_class)} "
                f"auto_tile={tile_size:g}m "
                f"xy_extent={input_type_info['xy_extent_m']:.2f}m"
            )
        elif args.verbose:
            print(
                "[INFO] Single-tree tile override: "
                f"tile_size_m={tile_size:g}"
            )

    base_keep_mask = np.ones(n_all, dtype=bool)
    denoise_removed_mask = np.zeros(n_all, dtype=bool)

    if bool(args.denoise):
        if old_cls_all is not None:
            binary_mask0 = (old_cls_all == 0) | (old_cls_all == 1)
            target_mask = binary_mask0.copy()
            if not bool(args.denoise_preserve_ground):
                target_mask |= (old_cls_all == int(args.ground_class_code))
        else:
            target_mask = np.ones(n_all, dtype=bool)

        xyz_target = xyz_all[target_mask]
        if xyz_target.shape[0] > 0:
            rad_name = _choose_radiometric_attribute(cloud)
            if rad_name is not None:
                attr_all = np.asarray(getattr(cloud, rad_name), dtype=np.float32)
                attr_target = attr_all[target_mask]
                noise_local = _strict_noise_mask_radiometric(
                    xyz_target,
                    attr_target,
                    k=int(args.rad_sor_k),
                    nsigma=float(args.rad_sor_nsigma),
                )
                print(f"[INFO] Radiometric denoise field: {rad_name}")
            else:
                noise_local = _strict_noise_mask_xyz(
                    xyz_target,
                    k=int(args.xyz_sor_k),
                    zscore_thr=float(args.xyz_sor_nsigma),
                )
                print("[INFO] Radiometric field unavailable or constant. Falling back to xyz-only strict SOR.")

            denoise_removed_mask[target_mask] = noise_local
            base_keep_mask = ~denoise_removed_mask
            print(f"[INFO] Denoise removed: {int(denoise_removed_mask.sum())} / {n_all} points")
        else:
            print("[INFO] Denoise requested, but no eligible points found.")

    ground_mask = np.zeros(n_all, dtype=bool)
    if old_cls_all is not None:
        ground_mask = (old_cls_all == int(args.ground_class_code))

    predict_mask = base_keep_mask.copy()

    if bool(args.exclude_ground_class):
        if old_cls_all is not None:
            if np.any(ground_mask):
                predict_mask &= ~ground_mask
                print(
                    f"[INFO] Ground exclusion enabled: class={int(args.ground_class_code)} | "
                    f"excluded {int((ground_mask & base_keep_mask).sum())} / {n_all} points from prediction"
                )
            else:
                print(
                    f"[INFO] Ground exclusion enabled, but class={int(args.ground_class_code)} "
                    f"not found in Classification. Running on all non-denoised points."
                )
        else:
            print(
                "[INFO] Ground exclusion enabled, but Classification not available. "
                "Running on all non-denoised points."
            )

    xyz = xyz_all[predict_mask]
    n = int(xyz.shape[0])

    if n == 0:
        raise RuntimeError("No points left for prediction after denoise/ground exclusion.")

    geom_selected_all = None
    geom_select = feat_cfg.get("geom_select", []) or []
    geom_select = list(geom_select) if isinstance(geom_select, (list, tuple)) else []
    log_omnivariance = bool(feat_cfg.get("log_omnivariance", False))

    if bool(feat_cfg.get("include_geom_features", False)) and args.geom_cache == "all":
        geom_k = _infer_geom_k(feat_cfg)
        if len(geom_select) == 0:
            raise RuntimeError("geom_cache=all expects geom_select to be set to keep in_dim consistent.")
        print(f"[INFO] Precomputing geom features ONCE for prediction points only (k={geom_k}) ...")
        t0 = _time.time()
        geom10_all = normals_and_eigenfeatures(xyz, k=int(geom_k))
        geom_selected_all = _geom_select(geom10_all, geom_select, log_omnivariance)
        print(f"[INFO] Geom precompute done in {(_time.time() - t0) / 60:.2f} min. geom_selected_all={geom_selected_all.shape}")

    model = build_model(model_cfg, in_dim).to(device)
    model.eval()

    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[WARN] strict=True load failed: {e}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[WARN] Loaded with strict=False. missing={len(missing)} unexpected={len(unexpected)}")

    logits_sum = np.zeros((n, num_classes), dtype=np.float32)
    weights_sum = np.zeros((n,), dtype=np.float32)

    p_leaf_sum = np.zeros((n,), dtype=np.float32)
    p_leaf_sq_sum = np.zeros((n,), dtype=np.float32)
    vote_count = np.zeros((n,), dtype=np.float32)

    # Diagnostic only: number of distinct offset layouts in which each
    # point received at least one model evaluation. Unlike vote_count,
    # this cannot be inflated by padding duplicates inside a chunk.
    layout_vote_count = np.zeros((n,), dtype=np.uint16)

    geom_woody_sum = np.zeros((n,), dtype=np.float32)
    geom_woody_wsum = np.zeros((n,), dtype=np.float32)

    sx = sy = float(tile_size)
    offsets = _make_offsets(votes_req, sx=sx, sy=sy, mode=args.vote_mode, rng=rng)

    batch_pos: List[np.ndarray] = []
    batch_x: List[np.ndarray] = []
    batch_map: List[np.ndarray] = []
    batch_geom: List[Optional[np.ndarray]] = []
    batch_spw: List[Optional[np.ndarray]] = []
    batch_tile_q: List[float] = []

    total_blocks = 0
    total_tiles_seen = 0
    total_tiles_rejected = 0
    t_start = _time.time()

    def _softmax_logits(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        m = logits.max(axis=-1, keepdims=True)
        ex = np.exp(logits - m)
        return ex / np.maximum(ex.sum(axis=-1, keepdims=True), 1e-12)

    def _entropy_weight_from_logits(logits_bnc: np.ndarray) -> np.ndarray:
        p = _softmax_logits(logits_bnc)
        ent = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=2)
        ent = ent / np.log(float(num_classes) + 1e-12)
        w = 1.0 - ent
        return np.clip(w, 0.2, 1.0).astype(np.float32)

    def _geom_vote_weight(geom_sel: np.ndarray) -> Optional[np.ndarray]:
        if geom_sel is None:
            return None
        name_to_idx = {name: i for i, name in enumerate(geom_select)}
        if ("linearity" not in name_to_idx) or ("scattering" not in name_to_idx):
            return None
        lin = geom_sel[:, name_to_idx["linearity"]].astype(np.float32)
        sca = geom_sel[:, name_to_idx["scattering"]].astype(np.float32)
        beta = float(args.geom_vote_beta)
        gmin = float(args.geom_vote_min)
        gmax = float(args.geom_vote_max)
        g = 0.5 + beta * (lin - sca)
        return np.clip(g, gmin, gmax).astype(np.float32)

    def _geom_woody_support(geom_sel: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if geom_sel is None:
            return None
        name_to_idx = {name: i for i, name in enumerate(geom_select)}
        if ("linearity" not in name_to_idx) or ("scattering" not in name_to_idx):
            return None
        lin = geom_sel[:, name_to_idx["linearity"]].astype(np.float32)
        sca = geom_sel[:, name_to_idx["scattering"]].astype(np.float32)
        sup = 0.5 + 0.5 * (lin - sca)
        return np.clip(sup, 0.0, 1.0).astype(np.float32)

    def _spatial_vote_weight(pts_global: np.ndarray) -> np.ndarray:
        xy = pts_global[:, :2].astype(np.float32)
        c = xy.mean(axis=0, keepdims=True)
        d = np.linalg.norm(xy - c, axis=1)
        sigma = float(max(1e-6, args.spatial_sigma_frac * tile_size))
        w = np.exp(-(d * d) / (2.0 * sigma * sigma)).astype(np.float32)
        return np.clip(w, float(args.spatial_min), 1.0).astype(np.float32)

    def flush():
        nonlocal total_blocks
        if not batch_pos:
            return

        pos_np = np.stack(batch_pos, axis=0).astype(np.float32)
        x_np = np.stack(batch_x, axis=0).astype(np.float32)

        pos_t = torch.from_numpy(pos_np).to(device=device, dtype=torch.float32)
        x_t = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)

        with _autocast(bool(args.amp)):
            out = model(pos_t, x_t)

        out = out.detach().float().cpu().numpy()
        if out.ndim == 3 and out.shape[1] == num_classes:
            logits_bnc = np.transpose(out, (0, 2, 1))
        else:
            logits_bnc = out

        prob_bnc = _softmax_logits(logits_bnc)

        if args.vote_weight == "confidence":
            w_bn = _entropy_weight_from_logits(logits_bnc)
        else:
            w_bn = np.ones((logits_bnc.shape[0], logits_bnc.shape[1]), dtype=np.float32)

        for b in range(logits_bnc.shape[0]):
            q = float(batch_tile_q[b])
            w_bn[b] *= np.clip(q, 0.25, 1.0)

        if geom_selected_all is not None:
            for b in range(logits_bnc.shape[0]):
                gsel = batch_geom[b]
                if gsel is None:
                    continue
                gw = _geom_vote_weight(gsel)
                if gw is not None:
                    w_bn[b] *= gw.reshape(-1).astype(np.float32)

        if bool(args.spatial_vote):
            for b in range(logits_bnc.shape[0]):
                sw = batch_spw[b]
                if sw is not None:
                    w_bn[b] *= sw.reshape(-1).astype(np.float32)

        for b in range(logits_bnc.shape[0]):
            idx_map = batch_map[b].astype(np.int64).reshape(-1)
            w = w_bn[b].reshape(-1).astype(np.float32)

            np.add.at(logits_sum, idx_map, logits_bnc[b] * w[:, None])
            np.add.at(weights_sum, idx_map, w)

            if num_classes >= 2:
                p_leaf_b = prob_bnc[b, :, 1].astype(np.float32)
                np.add.at(p_leaf_sum, idx_map, p_leaf_b)
                np.add.at(p_leaf_sq_sum, idx_map, p_leaf_b * p_leaf_b)
                np.add.at(vote_count, idx_map, np.ones_like(p_leaf_b, dtype=np.float32))

            gsel = batch_geom[b]
            if gsel is not None:
                gsup = _geom_woody_support(gsel)
                if gsup is not None:
                    np.add.at(geom_woody_sum, idx_map, gsup)
                    np.add.at(geom_woody_wsum, idx_map, np.ones_like(gsup, dtype=np.float32))

        batch_pos.clear()
        batch_x.clear()
        batch_map.clear()
        batch_geom.clear()
        batch_spw.clear()
        batch_tile_q.clear()
        total_blocks += len(pos_np)

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    for vi, off in enumerate(offsets, 1):
        print(f"[INFO] Vote {vi}/{len(offsets)} offset=({off[0]:.4f}, {off[1]:.4f}) ... building tiles")

        # Tracks unique point coverage for this layout only.
        layout_seen = np.zeros((n,), dtype=bool)

        groups = _tile_group_indices(xyz[:, :2], sx=sx, sy=sy, offset=off)

        order_tiles = np.arange(len(groups))
        rng.shuffle(order_tiles)

        if tqdm is not None and args.progress == "tiles":
            tile_iter = tqdm(order_tiles, desc=f"vote{vi} tiles", leave=True)
        else:
            tile_iter = order_tiles

        for ti, gi in enumerate(tile_iter, 1):
            tile_idx = groups[int(gi)]
            total_tiles_seen += 1

            if tile_idx.size < int(min_points):
                total_tiles_rejected += 1
                continue

            tile_pts = xyz[tile_idx]
            tile_q_score = 1.0
            if bool(args.adaptive_tile_checks):
                ok_tile, tile_metrics = _tile_quality_checks(
                    pts_global=tile_pts,
                    tile_size=float(tile_size),
                    min_points=int(min_points),
                    occupancy_grid_n=int(args.occupancy_grid_n),
                    min_occupancy_frac=float(args.min_occupancy_frac),
                    max_dominant_cell_frac=float(args.max_dominant_cell_frac),
                    min_xy_spread_frac=float(args.min_xy_spread_frac),
                )
                tile_q_score = float(tile_metrics.get("quality_score", 1.0))
                if not ok_tile:
                    total_tiles_rejected += 1
                    continue

            chunks = _chunk_no_replacement(tile_idx, n_points=int(n_points), rng=rng)
            for ch in chunks:
                # Boolean assignment intentionally collapses padding
                # duplicates, so each point contributes at most one
                # coverage vote for this offset layout.
                layout_seen[ch] = True

                pts = xyz[ch].copy()
                pts_centered = pts - pts.mean(axis=0, keepdims=True)

                geom_chunk = None
                if geom_selected_all is not None:
                    geom_chunk = geom_selected_all[ch]

                x_feat = build_features_for_block(
                    pts_centered=pts_centered,
                    pts_global=pts,
                    feat_cfg=feat_cfg,
                    expect_dim=in_dim,
                    geom_cached_selected=geom_chunk,
                )

                spw = _spatial_vote_weight(pts) if bool(args.spatial_vote) else None

                batch_pos.append(pts_centered.astype(np.float32))
                batch_x.append(x_feat.astype(np.float32))
                batch_map.append(ch.astype(np.int64))
                batch_geom.append(geom_chunk)
                batch_spw.append(spw)
                batch_tile_q.append(tile_q_score)

                if len(batch_pos) >= max(1, int(batch_blocks)):
                    flush()

            if tqdm is None and args.progress == "tiles" and (ti % int(args.progress_every_tiles) == 0):
                dt = _time.time() - t_start
                print(
                    f"[INFO] vote{vi}: tiles {ti}/{len(order_tiles)} | "
                    f"blocks={total_blocks} | rejected={total_tiles_rejected}/{total_tiles_seen} | "
                    f"elapsed={dt/60:.1f} min"
                )

        flush()

        # Count this offset once for every point actually evaluated in it.
        layout_vote_count += layout_seen.astype(np.uint16)

        layout_seen_n = int(layout_seen.sum())
        print(
            f"[INFO] Vote {vi} done. blocks={total_blocks} elapsed={(_time.time() - t_start) / 60:.1f} min | "
            f"rejected_tiles={total_tiles_rejected}/{total_tiles_seen} | "
            f"unique_points_seen={layout_seen_n}/{n} "
            f"({100.0 * layout_seen_n / max(1, n):.2f}%)"
        )

    if not np.any(weights_sum > 0):
        raise RuntimeError("No points received predictions. Something is wrong with tiling/chunking or tile qualification.")

    target_layouts = int(len(offsets))
    layout_coverage_stats = {
        "target_layouts": target_layouts,
        "mean": float(layout_vote_count.mean()) if n > 0 else 0.0,
        "median": float(np.median(layout_vote_count)) if n > 0 else 0.0,
        "min": int(layout_vote_count.min()) if n > 0 else 0,
        "max": int(layout_vote_count.max()) if n > 0 else 0,
        "pct_eq_target": float(100.0 * np.mean(layout_vote_count == target_layouts)) if n > 0 else 0.0,
        "pct_ge_target_minus_1": float(
            100.0 * np.mean(layout_vote_count >= max(1, target_layouts - 1))
        ) if n > 0 else 0.0,
        "pct_ge_target_minus_2": float(
            100.0 * np.mean(layout_vote_count >= max(1, target_layouts - 2))
        ) if n > 0 else 0.0,
    }

    print(
        "[INFO] Unique-layout coverage: "
        f"mean={layout_coverage_stats['mean']:.2f}/{target_layouts} "
        f"median={layout_coverage_stats['median']:.1f} "
        f"min={layout_coverage_stats['min']} "
        f"max={layout_coverage_stats['max']} | "
        f"exact={layout_coverage_stats['pct_eq_target']:.2f}% "
        f">={max(1, target_layouts - 1)}={layout_coverage_stats['pct_ge_target_minus_1']:.2f}% "
        f">={max(1, target_layouts - 2)}={layout_coverage_stats['pct_ge_target_minus_2']:.2f}%"
    )

    logits_avg = logits_sum / np.maximum(1e-6, weights_sum)[:, None]
    p = _softmax_logits(logits_avg)

    if num_classes != 2:
        pred = np.argmax(p, axis=1).astype(np.uint8)
        p_leaf = p[:, 1].astype(np.float32) if p.shape[1] > 1 else np.zeros((n,), dtype=np.float32)
        vote_var = np.zeros((n,), dtype=np.float32)
        decision_stats = {}
        local_woody_support = np.zeros((n,), dtype=np.float32)
        local_leaf_support = np.zeros((n,), dtype=np.float32)
        geom_woody_support = np.zeros((n,), dtype=np.float32)
    else:
        p_leaf = p[:, 1].astype(np.float32)

        vote_mean = p_leaf_sum / np.maximum(vote_count, 1.0)
        vote_var = np.maximum(0.0, p_leaf_sq_sum / np.maximum(vote_count, 1.0) - vote_mean * vote_mean).astype(np.float32)

        geom_woody_support = geom_woody_sum / np.maximum(geom_woody_wsum, 1.0)
        geom_woody_support = np.clip(geom_woody_support, 0.0, 1.0).astype(np.float32)

        probability_distribution_stats = None

        if args.mode == "raw":
            # Direct binary decision from the model leaf probability.
            # With --votes 1 this is the single-pass PT baseline.
            local_woody_support = np.zeros((n,), dtype=np.float32)
            local_leaf_support = np.zeros((n,), dtype=np.float32)
            pred = (p_leaf >= float(args.thr)).astype(np.uint8)
            decision_stats = {
                "mode": "raw_probability_threshold",
                "threshold": float(args.thr),
                "wood": int((pred == 0).sum()),
                "leaf": int((pred == 1).sum()),
            }
            print(f"[INFO] Raw decision stats: {decision_stats}")

        elif args.mode == "adaptive":
            probability_distribution_stats = _analyze_leaf_probability_distribution(
                p_leaf=p_leaf,
                bins=int(args.adaptive_hist_bins),
                smooth_sigma=float(args.adaptive_hist_smooth_sigma),
                shoulder_fraction=float(args.adaptive_shoulder_fraction),
                min_transition_width=float(args.adaptive_min_transition_width),
            )

            print(
                "[INFO] Adaptive probability distribution: "
                f"wood_peak={probability_distribution_stats['wood_peak']:.4f} "
                f"transition_low={probability_distribution_stats['transition_low']:.4f} "
                f"valley={probability_distribution_stats['valley']:.4f} "
                f"transition_high={probability_distribution_stats['transition_high']:.4f} "
                f"leaf_peak={probability_distribution_stats['leaf_peak']:.4f}"
            )
            print(
                "[INFO] Adaptive zones: "
                f"wood_anchor={100.0*probability_distribution_stats['wood_anchor_frac']:.2f}% "
                f"transition={100.0*probability_distribution_stats['transition_frac']:.2f}% "
                f"leaf_anchor={100.0*probability_distribution_stats['leaf_anchor_frac']:.2f}%"
            )

            if bool(args.adaptive_write_distribution):
                csv_path, svg_path = _write_probability_distribution_artifacts(
                    probability_distribution_stats,
                    args.out_las,
                )
                print(f"[INFO] Probability distribution CSV: {csv_path}")
                print(f"[INFO] Probability distribution SVG: {svg_path}")

            if geom_selected_all is None:
                raise RuntimeError(
                    "Adaptive mode requires precomputed selected geometric features."
                )

            low = float(probability_distribution_stats["transition_low"])
            high = float(probability_distribution_stats["transition_high"])
            strong_woody_anchor = p_leaf <= low
            strong_leaf_anchor = p_leaf >= high

            local_woody_support = _compute_local_woody_support(
                xyz=xyz,
                strong_woody_mask=strong_woody_anchor,
                k=int(args.midband_knn_k),
            )
            local_leaf_support = _compute_local_leaf_support(
                xyz=xyz,
                strong_leaf_mask=strong_leaf_anchor,
                k=int(args.midband_knn_k),
            )

            pred, decision_stats = _resolve_transition_adaptive(
                p_leaf=p_leaf,
                geom=geom_selected_all,
                analysis=probability_distribution_stats,
                local_woody_support=local_woody_support,
                local_leaf_support=local_leaf_support,
                vote_var=vote_var,
                vote_count=vote_count,
                geom_ratio=float(args.adaptive_geom_ratio),
                local_support_min=float(args.adaptive_local_support_min),
            )
            print(f"[INFO] Adaptive decision stats: {decision_stats}")

        else:
            strong_woody_anchor = p_leaf <= float(args.t_low)
            strong_leaf_anchor = p_leaf >= float(args.t_high)

            local_woody_support = _compute_local_woody_support(
                xyz=xyz,
                strong_woody_mask=strong_woody_anchor,
                k=int(args.midband_knn_k),
            )
            local_leaf_support = _compute_local_leaf_support(
                xyz=xyz,
                strong_leaf_mask=strong_leaf_anchor,
                k=int(args.midband_knn_k),
            )

            pred, decision_stats = _final_decision_three_zone(
                p_leaf=p_leaf,
                geom_woody_support=geom_woody_support,
                local_woody_support=local_woody_support,
                local_leaf_support=local_leaf_support,
                vote_var=vote_var,
                vote_count=vote_count,
                t_low=float(args.t_low),
                t_high=float(args.t_high),
                geom_rescue_thr=float(args.geom_rescue_thr),
                local_woody_thr=float(args.local_woody_thr),
                local_leaf_thr=float(args.local_leaf_thr),
                vote_var_thr=float(args.vote_var_thr),
                vote_count_low=float(args.vote_count_low),
            )

            print(f"[INFO] 3-zone decision stats: {decision_stats}")

    if bool(args.smooth) and num_classes == 2:
        t0 = _time.time()
        pred2, nflip = _knn_smooth_uncertain_only(
            xyz=xyz,
            cls=pred,
            p_leaf=p_leaf,
            vote_var=vote_var,
            vote_count=vote_count,
            k=int(args.smooth_k),
            tau=float(args.smooth_tau),
            margin=float(args.smooth_margin),
            passes=int(args.smooth_passes),
        )
        dt = _time.time() - t0
        print(f"[INFO] Smoothing done: flips={nflip} ({(100.0*nflip/max(1,n)):.4f}%) time={dt:.2f}s")
        pred = pred2


    woody_refine_flips = 0
    woody_structure_flips = 0
    woody_component_flips = 0
    woody_small_components = 0
    uncertain_reassign_to_woody = 0
    uncertain_reassign_to_leaf = 0
    core_woody_points = 0
    uncertain_points_total = 0
    uncertain_mask = np.zeros((n,), dtype=bool)

    if bool(args.post_refine_woody) and num_classes == 2:
        t0 = _time.time()
        pred2, woody_refine_flips = _post_refine_woody_points(
            xyz=xyz,
            cls=pred,
            p_leaf=p_leaf,
            k=int(args.woody_refine_k),
            min_woody_support=float(args.woody_refine_min_support),
            keep_if_p_leaf_le=float(args.woody_refine_keep_if_p_leaf_le),
            flip_only_if_p_leaf_ge=float(args.woody_refine_flip_only_if_p_leaf_ge),
        )
        dt = _time.time() - t0
        print(f"[INFO] Woody refine done: flips={woody_refine_flips} ({(100.0*woody_refine_flips/max(1,n)):.4f}%) time={dt:.2f}s")
        pred = pred2

    if num_classes == 2:
        core_woody_mask = _make_core_woody_mask(
            pred=pred,
            p_leaf=p_leaf,
            geom_woody_support=geom_woody_support,
            local_woody_support=local_woody_support,
            vote_var=vote_var,
            vote_count=vote_count,
            core_p_leaf_max=float(args.woody_core_p_leaf_max),
            core_min_local_support=float(args.woody_core_min_local_support),
            core_min_geom_support=float(args.woody_core_min_geom_support),
            core_max_vote_var=float(args.woody_core_max_vote_var),
            core_min_vote_count=float(args.woody_core_min_vote_count),
        )
        core_woody_points = int(core_woody_mask.sum())
        print(f"[INFO] Core woody points: {core_woody_points} / {n}")

        if bool(args.post_refine_woody_structure):
            t0 = _time.time()
            pred2, woody_structure_flips, uncertain1 = _post_refine_woody_structure(
                xyz=xyz,
                cls=pred,
                p_leaf=p_leaf,
                core_woody_mask=core_woody_mask,
                geom_woody_support=geom_woody_support,
                local_woody_support=local_woody_support,
                k=int(args.woody_structure_k),
                min_core_neighbor_frac=float(args.woody_structure_min_core_neighbor_frac),
                weak_p_leaf_ge=float(args.woody_structure_weak_p_leaf_ge),
                keep_if_geom_ge=float(args.woody_structure_keep_if_geom_ge),
                keep_if_local_ge=float(args.woody_structure_keep_if_local_ge),
            )
            dt = _time.time() - t0
            pred = pred2
            uncertain_mask |= uncertain1
            print(f"[INFO] Woody structure refine done: flips={woody_structure_flips} ({(100.0*woody_structure_flips/max(1,n)):.4f}%) time={dt:.2f}s")

            core_woody_mask = _make_core_woody_mask(
                pred=pred,
                p_leaf=p_leaf,
                geom_woody_support=geom_woody_support,
                local_woody_support=local_woody_support,
                vote_var=vote_var,
                vote_count=vote_count,
                core_p_leaf_max=float(args.woody_core_p_leaf_max),
                core_min_local_support=float(args.woody_core_min_local_support),
                core_min_geom_support=float(args.woody_core_min_geom_support),
                core_max_vote_var=float(args.woody_core_max_vote_var),
                core_min_vote_count=float(args.woody_core_min_vote_count),
            )
            core_woody_points = int(core_woody_mask.sum())
            print(f"[INFO] Core woody points (updated): {core_woody_points} / {n}")

        if bool(args.cleanup_small_woody_components):
            cleanup_candidate_mask = (pred == 0) & (~core_woody_mask)
            cleanup_candidate_count = int(cleanup_candidate_mask.sum())
            total_woody_now = int((pred == 0).sum())
            print(
                f"[INFO] Woody cleanup workload: total_woody={total_woody_now} "
                f"core={int(core_woody_mask.sum())} candidates={cleanup_candidate_count}"
            )
            t0 = _time.time()
            pred2, woody_component_flips, woody_small_components, uncertain2 = _cleanup_small_woody_components(
                xyz=xyz,
                cls=pred,
                p_leaf=p_leaf,
                core_woody_mask=core_woody_mask,
                radius=float(args.woody_component_radius),
                min_component_size=int(args.woody_component_min_size),
                keep_component_if_core_frac_ge=float(args.woody_component_keep_if_core_frac_ge),
                keep_component_if_mean_p_leaf_le=float(args.woody_component_keep_if_mean_p_leaf_le),
                cleanup_mask=cleanup_candidate_mask,
                max_candidates=int(args.woody_component_max_candidates),
                progress_every=int(args.woody_component_progress_every),
            )
            dt = _time.time() - t0
            pred = pred2
            uncertain_mask |= uncertain2
            print(f"[INFO] Woody component cleanup done: flips={woody_component_flips} small_components={woody_small_components} time={dt:.2f}s")

        if bool(args.reassign_uncertain) and np.any(uncertain_mask):
            t0 = _time.time()
            strong_leaf_mask = p_leaf >= float(args.t_high)
            pred2, uncertain_reassign_to_woody, uncertain_reassign_to_leaf = _reassign_uncertain_points(
                xyz=xyz,
                cls=pred,
                uncertain_mask=uncertain_mask,
                p_leaf=p_leaf,
                core_woody_mask=core_woody_mask,
                strong_leaf_mask=strong_leaf_mask,
                k=int(args.uncertain_reassign_k),
                woody_neighbor_frac_thr=float(args.uncertain_reassign_woody_neighbor_frac),
                leaf_neighbor_frac_thr=float(args.uncertain_reassign_leaf_neighbor_frac),
                woody_p_leaf_max=float(args.uncertain_reassign_woody_p_leaf_max),
            )
            dt = _time.time() - t0
            pred = pred2
            print(f"[INFO] Uncertain reassignment done: to_woody={uncertain_reassign_to_woody} to_leaf={uncertain_reassign_to_leaf} time={dt:.2f}s")

        uncertain_points_total = int(uncertain_mask.sum())

    if old_cls_all is not None:
        pred_full = old_cls_all.copy().astype(np.uint8)
    else:
        pred_full = np.zeros(n_all, dtype=np.uint8)

    p_leaf_full = np.full(n_all, float(args.ground_prob_fill), dtype=np.float32)
    p_leaf_full[predict_mask] = p_leaf

    if old_cls_all is not None and resolved_input_type == "plot":
        old_cls_all_u8 = old_cls_all.astype(np.uint8)

        is_ground = (old_cls_all_u8 == int(args.ground_class_code))
        is_binary = (old_cls_all_u8 == 0) | (old_cls_all_u8 == 1)

        pred_idx = np.where(predict_mask)[0]
        binary_pred_local = is_binary[predict_mask]
        if np.any(binary_pred_local):
            pred_full[pred_idx[binary_pred_local]] = pred[binary_pred_local]

        if np.any(is_ground):
            pred_full[is_ground] = np.uint8(args.ground_class_code)
            p_leaf_full[is_ground] = float(args.ground_prob_fill)

        preserve_mask = ~(is_binary | is_ground)
        if np.any(preserve_mask):
            pred_full[preserve_mask] = old_cls_all_u8[preserve_mask]
            p_leaf_full[preserve_mask] = float(args.ground_prob_fill)

        skipped_binary = is_binary & (~predict_mask)
        if np.any(skipped_binary):
            pred_full[skipped_binary] = old_cls_all_u8[skipped_binary]
            p_leaf_full[skipped_binary] = float(args.ground_prob_fill)

    else:
        # No-ground / isolated-tree semantics: all predicted points receive
        # the binary model result regardless of their incoming LAS class code.
        pred_full[predict_mask] = pred

    os.makedirs(os.path.dirname(os.path.abspath(args.out_las)) or ".", exist_ok=True)

    write_las_with_classification(
        cloud_or_las=cloud,
        out_path=args.out_las,
        pred_labels=pred_full,
        pred_leaf_prob=p_leaf_full,
        overwrite_classification=bool(args.overwrite_classification),
        save_pred_class_dim=bool(args.save_pred_class_dim),
        save_pred_prob_dim=bool(args.save_pred_prob_dim),
    )

    u, c = np.unique(pred_full, return_counts=True)
    print(f"[DONE] Wrote: {args.out_las}")
    print(f"[DONE] Pred dist: {dict(zip(u.tolist(), c.tolist()))}")

    changed_binary = None
    changed_vs_input = None
    if old_cls_all is not None:
        changed_binary = int(np.sum((old_cls_all != pred_full) & ((old_cls_all == 0) | (old_cls_all == 1))))
        print(f"[DONE] Binary classes updated: {changed_binary} points")
        changed_vs_input = int(np.sum(old_cls_all != pred_full))
        print(f"[DONE] Classification overwritten. Changed points vs input Classification: {changed_vs_input} / {n_all} ({100.0*changed_vs_input/max(1,n_all):.2f}%)")

    try:
        side = {
            "version": "1.0.1",
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "in_las": args.in_las,
            "out_las": args.out_las,
            "ckpt": args.ckpt,
            "config": args.config,
            "device": str(device),
            "tile_size_m": float(tile_size),
            "n_points": int(n_points),
            "votes": int(votes_req),
            "vote_mode": str(args.vote_mode),
            "vote_weight": str(args.vote_weight),
            "layout_coverage": layout_coverage_stats,
            "adaptive_probability_distribution": (
                None
                if probability_distribution_stats is None
                else {
                    k: v
                    for k, v in probability_distribution_stats.items()
                    if k not in ("centers", "hist", "smooth")
                }
            ),
            "batch_blocks": int(batch_blocks),
            "min_points": int(min_points),
            "geom_cache": str(args.geom_cache),
            "spatial_vote": bool(args.spatial_vote),
            "spatial_sigma_frac": float(args.spatial_sigma_frac),
            "spatial_min": float(args.spatial_min),
            "smooth": bool(args.smooth),
            "smooth_k": int(args.smooth_k),
            "smooth_tau": float(args.smooth_tau),
            "smooth_margin": float(args.smooth_margin),
            "smooth_passes": int(args.smooth_passes),
            "adaptive_tile_checks": bool(args.adaptive_tile_checks),
            "occupancy_grid_n": int(args.occupancy_grid_n),
            "min_occupancy_frac": float(args.min_occupancy_frac),
            "max_dominant_cell_frac": float(args.max_dominant_cell_frac),
            "min_xy_spread_frac": float(args.min_xy_spread_frac),
            "t_low": float(args.t_low),
            "t_high": float(args.t_high),
            "geom_rescue_thr": float(args.geom_rescue_thr),
            "local_woody_thr": float(args.local_woody_thr),
            "local_leaf_thr": float(args.local_leaf_thr),
            "midband_knn_k": int(args.midband_knn_k),
            "vote_var_thr": float(args.vote_var_thr),
            "vote_count_low": float(args.vote_count_low),
            "input_type_requested": str(input_type_info["requested"]),
            "input_type_resolved": str(input_type_info["resolved"]),
            "input_xy_extent_m": float(input_type_info["xy_extent_m"]),
            "input_ground_points": int(input_type_info["ground_points"]),
            "input_ground_fraction": float(input_type_info["ground_fraction"]),
            "input_usable_ground": bool(input_type_info["usable_ground"]),
            "single_tree_auto_tile": bool(
                resolved_input_type == "single_tree" and not tile_explicit
            ),
            "exclude_ground_class": bool(args.exclude_ground_class),
            "ground_class_code": int(args.ground_class_code),
            "ground_prob_fill": float(args.ground_prob_fill),
            "ground_points_excluded": int((ground_mask & base_keep_mask).sum()) if bool(args.exclude_ground_class) else 0,
            "denoise": bool(args.denoise),
            "denoise_preserve_ground": bool(args.denoise_preserve_ground),
            "rad_sor_k": int(args.rad_sor_k),
            "rad_sor_nsigma": float(args.rad_sor_nsigma),
            "xyz_sor_k": int(args.xyz_sor_k),
            "xyz_sor_nsigma": float(args.xyz_sor_nsigma),
            "denoise_removed_points": int(denoise_removed_mask.sum()),
            "post_refine_woody": bool(args.post_refine_woody),
            "woody_refine_k": int(args.woody_refine_k),
            "woody_refine_min_support": float(args.woody_refine_min_support),
            "woody_refine_keep_if_p_leaf_le": float(args.woody_refine_keep_if_p_leaf_le),
            "woody_refine_flip_only_if_p_leaf_ge": float(args.woody_refine_flip_only_if_p_leaf_ge),
            "woody_refine_flips": int(woody_refine_flips),
            "post_refine_woody_structure": bool(args.post_refine_woody_structure),
            "woody_structure_k": int(args.woody_structure_k),
            "woody_core_p_leaf_max": float(args.woody_core_p_leaf_max),
            "woody_core_min_local_support": float(args.woody_core_min_local_support),
            "woody_core_min_geom_support": float(args.woody_core_min_geom_support),
            "woody_core_max_vote_var": float(args.woody_core_max_vote_var),
            "woody_core_min_vote_count": float(args.woody_core_min_vote_count),
            "woody_structure_min_core_neighbor_frac": float(args.woody_structure_min_core_neighbor_frac),
            "woody_structure_weak_p_leaf_ge": float(args.woody_structure_weak_p_leaf_ge),
            "woody_structure_keep_if_geom_ge": float(args.woody_structure_keep_if_geom_ge),
            "woody_structure_keep_if_local_ge": float(args.woody_structure_keep_if_local_ge),
            "woody_structure_flips": int(woody_structure_flips),
            "cleanup_small_woody_components": bool(args.cleanup_small_woody_components),
            "woody_component_radius": float(args.woody_component_radius),
            "woody_component_min_size": int(args.woody_component_min_size),
            "woody_component_keep_if_core_frac_ge": float(args.woody_component_keep_if_core_frac_ge),
            "woody_component_keep_if_mean_p_leaf_le": float(args.woody_component_keep_if_mean_p_leaf_le),
            "woody_component_flips": int(woody_component_flips),
            "woody_small_components": int(woody_small_components),
            "reassign_uncertain": bool(args.reassign_uncertain),
            "uncertain_reassign_k": int(args.uncertain_reassign_k),
            "uncertain_reassign_woody_neighbor_frac": float(args.uncertain_reassign_woody_neighbor_frac),
            "uncertain_reassign_leaf_neighbor_frac": float(args.uncertain_reassign_leaf_neighbor_frac),
            "uncertain_reassign_woody_p_leaf_max": float(args.uncertain_reassign_woody_p_leaf_max),
            "uncertain_points_total": int(uncertain_points_total),
            "uncertain_reassign_to_woody": int(uncertain_reassign_to_woody),
            "uncertain_reassign_to_leaf": int(uncertain_reassign_to_leaf),
            "core_woody_points": int(core_woody_points),
            "predicted_points_only": int(n),
            "total_input_points": int(n_all),
            "decision_stats": decision_stats if num_classes == 2 else None,
            "overwrite_classification": bool(args.overwrite_classification),
            "save_pred_class_dim": bool(args.save_pred_class_dim),
            "save_pred_prob_dim": bool(args.save_pred_prob_dim),
            "pred_dist": {str(int(k)): int(v) for k, v in zip(u.tolist(), c.tolist())},
            "tiles_seen": int(total_tiles_seen),
            "tiles_rejected": int(total_tiles_rejected),
            "tiles_rejected_frac": float(total_tiles_rejected / max(1, total_tiles_seen)),
            "vote_count_mean": float(vote_count.mean()) if num_classes == 2 else None,
            "vote_var_mean": float(vote_var.mean()) if num_classes == 2 else None,
            "geom_woody_support_mean": float(geom_woody_support.mean()) if num_classes == 2 else None,
            "local_woody_support_mean": float(local_woody_support.mean()) if num_classes == 2 else None,
            "local_leaf_support_mean": float(local_leaf_support.mean()) if num_classes == 2 else None,
            "changed_binary": changed_binary,
            "changed_vs_input": changed_vs_input,
        }
        side_path = os.path.splitext(args.out_las)[0] + "_infer.json"
        with open(side_path, "w", encoding="utf-8") as f:
            json.dump(side, f, indent=2)
        print(f"[DONE] Sidecar: {side_path}")
    except Exception as e:
        print(f"[WARN] Sidecar write failed: {e}")

    return 0


def _collect_las_files(in_dir: str, recursive: bool = False) -> List[str]:
    exts = {".las", ".laz"}
    out: List[str] = []
    if recursive:
        for root, _, files in os.walk(in_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    out.append(os.path.join(root, f))
    else:
        for f in os.listdir(in_dir):
            p = os.path.join(in_dir, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in exts:
                out.append(p)
    out.sort()
    return out

def main(argv=None) -> int:
    argv = list(argv if argv is not None else [])
    if not argv:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--in_dir", "--in-dir", dest="in_dir", default=None)
    p.add_argument("--out_dir", "--out-dir", dest="out_dir", default=None)
    p.add_argument("--recursive", action="store_true", default=False)
    p.add_argument(
        "--skip_existing",
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--max_files",
        "--max-files",
        dest="max_files",
        type=int,
        default=None,
        help="Optional batch smoke-test limit; process only the first N sorted LAS/LAZ files.",
    )
    p.add_argument("--out_suffix", "--out-suffix", dest="out_suffix", default="_pointtransformer_pred")
    known, _ = p.parse_known_args(argv)

    if known.in_dir:
        in_dir = os.path.abspath(known.in_dir)
        if not os.path.isdir(in_dir):
            raise FileNotFoundError(f"Input folder not found: {in_dir}")

        out_dir = os.path.abspath(known.out_dir) if known.out_dir else (in_dir.rstrip("\\/") + str(known.out_suffix))
        os.makedirs(out_dir, exist_ok=True)
        files = _collect_las_files(in_dir, recursive=bool(known.recursive))
        if not files:
            raise FileNotFoundError(f"No .las or .laz files found in: {in_dir}")

        if known.max_files is not None:
            if int(known.max_files) <= 0:
                raise ValueError("--max_files must be > 0.")
            files = files[: int(known.max_files)]

        print("")
        print("points2SBL batch inference")
        print(f"Files  : {len(files)}")
        print(f"Input  : {in_dir}")
        print(f"Output : {out_dir}")

        passthrough = []
        skip_next = False
        value_flags = {
            "--in_dir", "--in-dir",
            "--out_dir", "--out-dir",
            "--out_suffix", "--out-suffix",
            "--max_files", "--max-files",
        }
        for tok in argv:
            if skip_next:
                skip_next = False
                continue
            if tok in value_flags:
                skip_next = True
                continue
            if tok in {"--recursive", "--skip_existing", "--skip-existing"}:
                continue
            passthrough.append(tok)

        failures = 0
        skipped = 0
        for i, f in enumerate(files, 1):
            rel = os.path.relpath(f, in_dir)
            out_file = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

            if bool(known.skip_existing) and os.path.exists(out_file):
                skipped += 1
                print(f"[SKIP {i}/{len(files)}] Output exists: {out_file}")
                continue

            per_file_argv = passthrough + ["--in_las", f, "--out_las", out_file]
            print("=" * 60)
            print(f"[FILE {i}/{len(files)}] {f}")
            print(f"[OUT ] {out_file}")
            print("=" * 60)
            try:
                rc = int(_predict_cli_main(per_file_argv))
            except Exception as e:
                failures += 1
                print(f"[ERROR] Prediction failed for {f}: {type(e).__name__}: {e}")
                continue
            if rc != 0:
                failures += 1

        print("")
        print("Batch complete")
        print(f"Files   : {len(files)}")
        print(f"Skipped : {skipped}")
        print(f"Failed  : {failures}")
        print(f"Output  : {out_dir}")

        if failures:
            return 1
        return 0

    return int(_predict_cli_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
