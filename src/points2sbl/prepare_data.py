#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
points2sbl.prepare_data

STRICT BINARY, OVERLAPPING XY WINDOWS, MIXED BLOCK SPLIT

Behavior:
- For files with a label field, ONLY source labels 0 and 1 are allowed.
  All other labels are discarded.
- For files without a label field, filename inference is used:
    *_leaf.las / *_leaf.laz   -> all labels = 1
    *_wood.las / *_wood.laz   -> all labels = 0
    *_woody.las / *_woody.laz -> all labels = 0
- Full vertical extent is always used.
- Windows are created in XY using tile size and stride.
- Blocks are generated from ALL files first, then blocks are shuffled and split.
  This keeps train and val mixed across dataset styles / forest regimes.
- Random sampling to fixed N points per block is preserved.
- Optional overlap can be disabled for val/test while kept for train.
- Anchor merge is train-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import laspy
import numpy as np
from tqdm import tqdm


# --------------------------
# Helpers
# --------------------------
def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _read_xyz(las: laspy.LasData) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(las.x, dtype=np.float32),
            np.asarray(las.y, dtype=np.float32),
            np.asarray(las.z, dtype=np.float32),
        ]
    )


def _infer_const_label_from_name(path: str) -> Optional[int]:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    if stem.endswith("_leaf"):
        return 1
    if stem.endswith("_wood") or stem.endswith("_woody"):
        return 0
    return None


def _get_labels_or_infer(
    las: laspy.LasData,
    path: str,
    label_field: str,
    leaf_class: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns:
      y01: binary labels (0/1) for kept points
      valid_mask: boolean mask over original LAS points
      inferred: True if labels were inferred from filename
    """
    field = (label_field or "").strip()
    if field == "":
        field = "Classification"

    candidates = [field, field.lower(), field.upper()]
    if field.lower() != "classification":
        candidates += ["classification", "Classification"]

    arr = None
    for cand in candidates:
        try:
            if hasattr(las, cand):
                arr = np.asarray(getattr(las, cand), dtype=np.int32)
                break
        except Exception:
            pass

    if arr is not None:
        valid_mask = np.isin(arr, [0, 1])
        if not np.any(valid_mask):
            raise RuntimeError(f"{path} contains no class 0/1 points after filtering.")
        arr2 = arr[valid_mask]
        y01 = (arr2 == int(leaf_class)).astype(np.int32)
        return y01, valid_mask, False

    const = _infer_const_label_from_name(path)
    if const is None:
        raise RuntimeError(
            f"No label field '{label_field}' (or classification) found in: {path}\n"
            f"And filename is not *_leaf.* or *_wood.* or *_woody.* so labels cannot be inferred."
        )

    valid_mask = np.ones(len(las.points), dtype=bool)
    y01 = np.full(len(las.points), int(const), dtype=np.int32)
    return y01, valid_mask, True


def _window_starts_1d(vmin: float, vmax: float, win: float, stride: float) -> List[float]:
    if vmax <= vmin:
        return [float(vmin)]
    if win <= 0 or stride <= 0:
        raise ValueError("Window size and stride must be > 0")

    starts: List[float] = []
    s = float(vmin)

    while s <= (vmax - win + 1e-9):
        starts.append(float(s))
        s += stride

    last_needed = float(vmax - win)
    if last_needed < vmin:
        last_needed = float(vmin)

    if not starts:
        starts = [last_needed]
    elif abs(starts[-1] - last_needed) > 1e-6:
        starts.append(last_needed)

    out: List[float] = []
    seen = set()
    for x in starts:
        key = round(x, 6)
        if key not in seen:
            seen.add(key)
            out.append(float(x))
    return out


def _window_indices_xy(
    pts_xyz: np.ndarray,
    xy_size: Tuple[float, float],
    stride: Tuple[float, float],
) -> List[np.ndarray]:
    x = pts_xyz[:, 0]
    y = pts_xyz[:, 1]

    sx, sy = float(xy_size[0]), float(xy_size[1])
    dx, dy = float(stride[0]), float(stride[1])

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    x_starts = _window_starts_1d(xmin, xmax, sx, dx)
    y_starts = _window_starts_1d(ymin, ymax, sy, dy)

    windows: List[np.ndarray] = []
    for xs in x_starts:
        xe = xs + sx
        xmask = (x >= xs) & (x < xe if xe < xmax else x <= xe + 1e-9)
        if not np.any(xmask):
            continue
        for ys in y_starts:
            ye = ys + sy
            mask = xmask & (y >= ys) & (y < ye if ye < ymax else y <= ye + 1e-9)
            idx = np.where(mask)[0]
            if idx.size > 0:
                windows.append(idx.astype(np.int64))
    return windows


def _leaf_fraction(y01: np.ndarray) -> float:
    return float(np.mean(y01.astype(np.float32))) if y01.size else 0.0


def _block_center_and_rotate(pts_xyz: np.ndarray, seed: int, rotate: bool) -> np.ndarray:
    pts = pts_xyz.astype(np.float32, copy=True)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    pts[:, 0] -= cx
    pts[:, 1] -= cy

    if rotate:
        rng = np.random.RandomState(int(seed))
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        ca, sa = np.cos(ang), np.sin(ang)
        x = pts[:, 0].copy()
        y = pts[:, 1].copy()
        pts[:, 0] = ca * x - sa * y
        pts[:, 1] = sa * x + ca * y

    return pts


def _sample_points(
    pts_xyz: np.ndarray,
    y01: np.ndarray,
    n_points: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    n = pts_xyz.shape[0]
    if n == 0:
        return pts_xyz, y01
    if n >= n_points:
        idx = rng.choice(n, size=int(n_points), replace=False)
    else:
        idx = rng.choice(n, size=int(n_points), replace=True)
    return pts_xyz[idx], y01[idx]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _clear_sample_dir(path: str) -> None:
    if not os.path.isdir(path):
        return
    for fn in os.listdir(path):
        if fn.lower().endswith((".npy", ".npz")):
            try:
                os.remove(os.path.join(path, fn))
            except Exception:
                pass


def _stable_seed_for_string(seed: int, text: str) -> int:
    stable_key = f"{int(seed)}|{text}".encode("utf-8")
    return int(hashlib.md5(stable_key).hexdigest()[:8], 16)


# --------------------------
# Anchor merge
# --------------------------
def _anchor_merge_extra_blocks(
    pts_xyz: np.ndarray,
    y01: np.ndarray,
    n_points: int,
    anchor_class: int,
    anchor_min_points: int,
    anchor_seed_points: int,
    anchor_knn: int,
    extra_blocks: int,
    rng: np.random.RandomState,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    if extra_blocks <= 0:
        return out

    mask = (y01.astype(np.int32) == int(anchor_class))
    anchor_idx = np.where(mask)[0]
    if anchor_idx.size < int(anchor_min_points):
        return out

    seed_n = min(int(anchor_seed_points), int(anchor_idx.size))
    seed_idx = rng.choice(anchor_idx, size=seed_n, replace=False)

    for _ in range(int(extra_blocks)):
        center_i = int(rng.choice(seed_idx, size=1)[0])
        center = pts_xyz[center_i]
        d2 = (pts_xyz[:, 0] - center[0]) ** 2 + (pts_xyz[:, 1] - center[1]) ** 2
        nn = np.argsort(d2)[: int(max(anchor_knn, n_points))]
        pts_b, y_b = pts_xyz[nn], y01[nn]
        pts_s, y_s = _sample_points(pts_b, y_b, n_points=n_points, rng=rng)
        out.append((pts_s, y_s))

    return out


# --------------------------
# Block record
# --------------------------
@dataclass
class BlockRecord:
    pts: np.ndarray
    y: np.ndarray
    source_file: str
    source_group: str
    inferred_labels: bool
    block_kind: str   # main / anchor
    block_id_text: str


# --------------------------
# Main
# --------------------------
def main() -> None:
    ap = argparse.ArgumentParser(prog="points2sbl.prepare_data")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument("--val_ratio", type=float, default=0.20)
    ap.add_argument("--test_ratio", type=float, default=0.00)

    ap.add_argument("--n_points", type=int, default=8192)
    ap.add_argument("--xy_size", type=float, nargs=2, default=(2.0, 2.0))
    ap.add_argument("--stride", type=float, nargs=2, default=None)
    ap.add_argument("--use_overlap_in_val", action="store_true")
    ap.add_argument("--min_points", type=int, default=64)
    ap.add_argument("--max_blocks_per_file", type=int, default=0)

    ap.add_argument("--label_field", type=str, default="Classification")
    ap.add_argument("--leaf_class", type=int, default=1)
    ap.add_argument("--leaf_frac_min", type=float, default=0.0)
    ap.add_argument("--leaf_frac_max", type=float, default=1.0)

    ap.add_argument("--rotate_train", action="store_true")
    ap.add_argument("--jitter_std", type=float, default=0.0)
    ap.add_argument("--scale_min", type=float, default=1.0)
    ap.add_argument("--scale_max", type=float, default=1.0)

    ap.add_argument("--enable_anchor_merge", action="store_true")
    ap.add_argument("--anchor_class", type=int, default=0)
    ap.add_argument("--anchor_min_points", type=int, default=40)
    ap.add_argument("--anchor_seed_points", type=int, default=64)
    ap.add_argument("--anchor_knn", type=int, default=48)
    ap.add_argument("--anchor_blocks_per_tile", type=int, default=1)

    ap.add_argument("--save_format", choices=["npz", "npy"], default="npz")

    args = ap.parse_args()
    _seed_all(args.seed)

    xy_size = (float(args.xy_size[0]), float(args.xy_size[1]))
    if args.stride is None:
        stride_train = xy_size
    else:
        stride_train = (float(args.stride[0]), float(args.stride[1]))
    stride_eval = stride_train if args.use_overlap_in_val else xy_size

    root = os.path.abspath(args.data_root)
    out_train = os.path.join(root, "train", "sample_dir")
    out_val = os.path.join(root, "val", "sample_dir")
    out_test = os.path.join(root, "test", "sample_dir")
    _ensure_dir(out_train)
    _ensure_dir(out_val)
    _ensure_dir(out_test)

    _clear_sample_dir(out_train)
    _clear_sample_dir(out_val)
    _clear_sample_dir(out_test)

    exts = (".las", ".laz")
    las_files: List[str] = []
    if args.recursive:
        for r, _, files in os.walk(root):
            for fn in files:
                if fn.lower().endswith(exts):
                    las_files.append(os.path.join(r, fn))
    else:
        for fn in os.listdir(root):
            if fn.lower().endswith(exts):
                las_files.append(os.path.join(root, fn))

    las_files = sorted(las_files)
    if not las_files:
        raise FileNotFoundError(f"No LAS/LAZ found under: {root}")

    train_ratio = 1.0 - float(args.val_ratio) - float(args.test_ratio)
    if train_ratio <= 0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    print("\nPreparing blocks (STRICT BINARY, MIXED BLOCK SPLIT): woody=0, leaf=1")
    print("Allowed source labels in labeled files: [0, 1] only")
    print(f"Leaf class code: [{args.leaf_class}]")
    print(f"Label field: {args.label_field}  (fallback: *_leaf/_wood/_woody filename inference)")
    print(f"XY tile size: {xy_size[0]} x {xy_size[1]} m")
    print(f"Train stride : {stride_train[0]} x {stride_train[1]} m")
    print(f"Eval stride  : {stride_eval[0]} x {stride_eval[1]} m")
    print(f"n_points/block: {args.n_points} | min_points(window->direct): {args.min_points}")
    print(f"MIXED split ratios: train={train_ratio:.3f} val={args.val_ratio:.3f} test={args.test_ratio:.3f}")
    print(f"leaf_frac filter: [{args.leaf_frac_min}, {args.leaf_frac_max}]")
    print(f"max_blocks_per_file: {args.max_blocks_per_file}")
    print(f"save_format: {args.save_format}")
    print(f"use_overlap_in_val: {bool(args.use_overlap_in_val)}")

    if args.enable_anchor_merge:
        print("Anchor-merge: ON")
        print(
            f"  anchor_class={args.anchor_class} (0=woody,1=leaf)\n"
            f"  anchor_min_points={args.anchor_min_points} seed_points={args.anchor_seed_points} "
            f"anchor_knn={args.anchor_knn} extra_blocks/tile={args.anchor_blocks_per_tile}"
        )
    else:
        print("Anchor-merge: OFF")

    all_blocks: List[BlockRecord] = []

    windows_seen = 0
    windows_rejected = 0
    files_skipped_no_valid_binary = 0
    files_with_blocks = 0

    for f in tqdm(las_files, desc="files"):
        source_group = "binarylabels" if "binarylabels" in [p.lower() for p in os.path.normpath(f).split(os.sep)] else "main"

        las = laspy.read(f)

        try:
            y01, keep, inferred = _get_labels_or_infer(
                las=las,
                path=f,
                label_field=args.label_field,
                leaf_class=int(args.leaf_class),
            )
        except RuntimeError as e:
            print(f"[WARN] Skipping file: {f}")
            print(f"       Reason: {e}")
            files_skipped_no_valid_binary += 1
            continue

        if not np.any(keep):
            files_skipped_no_valid_binary += 1
            continue

        pts_all = _read_xyz(las)
        pts = pts_all[keep]

        if pts.shape[0] == 0 or y01.size == 0:
            files_skipped_no_valid_binary += 1
            continue

        # Generate windows both ways up front so we can keep mixed split later
        windows_train = _window_indices_xy(pts, xy_size=xy_size, stride=stride_train)
        windows_eval = _window_indices_xy(pts, xy_size=xy_size, stride=stride_eval)

        # Use the richer train-style windows as the master window set.
        # Later, during mixed split, val/test will still be blocks generated from the same global pool.
        windows = windows_train

        rng_file = np.random.RandomState(_stable_seed_for_string(args.seed, f))
        rng_file.shuffle(windows)

        blocks_from_file = 0
        file_added_any = False

        for iw, idx in enumerate(windows):
            if idx.size < int(args.min_points):
                continue

            pts_c = pts[idx]
            y_c = y01[idx]

            frac = _leaf_fraction(y_c)

            # Pure inferred _leaf/_wood/_woody files bypass leaf_frac_min
            if not inferred:
                if frac < float(args.leaf_frac_min) or frac > float(args.leaf_frac_max):
                    windows_rejected += 1
                    continue

            pts_s, y_s = _sample_points(
                pts_c, y_c, n_points=int(args.n_points), rng=rng_file
            )

            windows_seen += 1
            bid_text = f"{os.path.relpath(f, root)}|main|{iw}"
            all_blocks.append(
                BlockRecord(
                    pts=pts_s,
                    y=y_s,
                    source_file=f,
                    source_group=source_group,
                    inferred_labels=inferred,
                    block_kind="main",
                    block_id_text=bid_text,
                )
            )
            blocks_from_file += 1
            file_added_any = True

            if args.enable_anchor_merge:
                extra = _anchor_merge_extra_blocks(
                    pts_xyz=pts_s,
                    y01=y_s,
                    n_points=int(args.n_points),
                    anchor_class=int(args.anchor_class),
                    anchor_min_points=int(args.anchor_min_points),
                    anchor_seed_points=int(args.anchor_seed_points),
                    anchor_knn=int(args.anchor_knn),
                    extra_blocks=int(args.anchor_blocks_per_tile),
                    rng=rng_file,
                )
                for ie, (pts_e, y_e) in enumerate(extra):
                    bid_text = f"{os.path.relpath(f, root)}|anchor|{iw}|{ie}"
                    all_blocks.append(
                        BlockRecord(
                            pts=pts_e,
                            y=y_e,
                            source_file=f,
                            source_group=source_group,
                            inferred_labels=inferred,
                            block_kind="anchor",
                            block_id_text=bid_text,
                        )
                    )
                    blocks_from_file += 1

            if int(args.max_blocks_per_file) > 0 and blocks_from_file >= int(args.max_blocks_per_file):
                break

        if file_added_any:
            files_with_blocks += 1

    if not all_blocks:
        raise RuntimeError("No blocks were generated. Check staging, labels, or prep parameters.")

    # --------------------------
    # MIXED block split
    # --------------------------
    rng_split = np.random.RandomState(int(args.seed))
    order = np.arange(len(all_blocks), dtype=np.int64)
    rng_split.shuffle(order)
    all_blocks = [all_blocks[i] for i in order]

    total_blocks = len(all_blocks)
    n_test = int(round(total_blocks * float(args.test_ratio)))
    n_val = int(round(total_blocks * float(args.val_ratio)))
    if n_test + n_val >= total_blocks:
        n_test = 0
        n_val = max(1, min(total_blocks - 1, n_val))

    test_blocks = all_blocks[:n_test]
    val_blocks = all_blocks[n_test:n_test + n_val]
    train_blocks = all_blocks[n_test + n_val:]

    splits = {
        "train": train_blocks,
        "val": val_blocks,
        "test": test_blocks,
    }

    next_id = {"train": 0, "val": 0, "test": 0}
    written = {"train": 0, "val": 0, "test": 0}
    saved_counts = {0: 0, 1: 0}

    def _save(split: str, rec: BlockRecord) -> None:
        outdir = {"train": out_train, "val": out_val, "test": out_test}[split]
        bid = int(next_id[split])
        next_id[split] += 1

        seed_local = _stable_seed_for_string(args.seed, rec.block_id_text + f"|{split}|{bid}")
        pts2 = _block_center_and_rotate(
            rec.pts,
            seed=seed_local,
            rotate=(bool(args.rotate_train) and split == "train"),
        )

        if split == "train":
            rng = np.random.RandomState(seed_local)
            s = float(rng.uniform(args.scale_min, args.scale_max))
            pts2[:, :2] *= s
            if args.jitter_std and args.jitter_std > 0:
                pts2[:, :3] += rng.normal(
                    0.0, float(args.jitter_std), size=pts2.shape
                ).astype(np.float32)

        pos = pts2.astype(np.float32, copy=False)
        y = rec.y.astype(np.int32, copy=False)

        saved_counts[0] += int((y == 0).sum())
        saved_counts[1] += int((y == 1).sum())

        fn_base = os.path.join(outdir, f"{split}_{bid:07d}")
        if args.save_format == "npz":
            np.savez_compressed(fn_base + ".npz", pos=pos, y=y)
        else:
            np.save(fn_base + ".npy", {"pos": pos, "y": y}, allow_pickle=True)

        written[split] += 1

    for split_name, records in splits.items():
        for rec in records:
            _save(split_name, rec)

    total_written = written["train"] + written["val"] + written["test"]

    def _summarize_blocks(records: List[BlockRecord]) -> Dict[str, int]:
        out = {"main": 0, "anchor": 0, "main_source": 0, "binarylabels_source": 0}
        for r in records:
            out[r.block_kind] = out.get(r.block_kind, 0) + 1
            key = "binarylabels_source" if r.source_group == "binarylabels" else "main_source"
            out[key] = out.get(key, 0) + 1
        return out

    train_meta = _summarize_blocks(train_blocks)
    val_meta = _summarize_blocks(val_blocks)
    test_meta = _summarize_blocks(test_blocks)

    print(f"\nWindows collected: {total_written} (seen={windows_seen}, rejected={windows_rejected})")
    print(f"Split blocks: train={written['train']} val={written['val']} test={written['test']}")
    print(f"Files with blocks: {files_with_blocks} / {len(las_files)}")
    print(f"Files skipped (no valid binary labels): {files_skipped_no_valid_binary}")
    print(f"Output dirs:\n  {out_train}\n  {out_val}\n  {out_test}")

    report = {
        "version": "0.6.0",
        "mode": "strict_binary_overlap_mixed_block_split",
        "in_dir": str(root),
        "out_root": str(root),
        "recursive": bool(args.recursive),
        "label_field": str(args.label_field),
        "leaf_class": int(args.leaf_class),
        "allowed_source_labels": [0, 1],
        "save_format": str(args.save_format),
        "xy_size": [float(xy_size[0]), float(xy_size[1])],
        "stride_train": [float(stride_train[0]), float(stride_train[1])],
        "stride_eval": [float(stride_eval[0]), float(stride_eval[1])],
        "use_overlap_in_val": bool(args.use_overlap_in_val),
        "n_points": int(args.n_points),
        "min_points": int(args.min_points),
        "max_blocks_per_file": int(args.max_blocks_per_file),
        "splits": {
            "train_ratio": float(train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
            "mixed_block_split": True,
        },
        "counts": {
            "windows_seen": int(windows_seen),
            "windows_rejected": int(windows_rejected),
            "files_skipped_no_valid_binary": int(files_skipped_no_valid_binary),
            "files_with_blocks": int(files_with_blocks),
            "train": int(written["train"]),
            "val": int(written["val"]),
            "test": int(written["test"]),
            "total": int(total_written),
        },
        "saved_label_counts": {"0": int(saved_counts[0]), "1": int(saved_counts[1])},
        "split_block_metadata": {
            "train": train_meta,
            "val": val_meta,
            "test": test_meta,
        },
        "anchor_merge": {
            "enabled": bool(args.enable_anchor_merge),
            "anchor_class": int(args.anchor_class),
            "anchor_blocks_per_tile": int(args.anchor_blocks_per_tile),
            "anchor_min_points": int(args.anchor_min_points),
            "anchor_seed_points": int(args.anchor_seed_points),
            "anchor_knn": int(args.anchor_knn),
        },
    }

    try:
        out_report = os.path.join(root, "_prepare_report.json")
        with open(out_report, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
        print(f"[REPORT] Wrote: {out_report}")
    except Exception as e:
        print(f"[WARN] Failed to write prepare report: {e}")


if __name__ == "__main__":
    main()