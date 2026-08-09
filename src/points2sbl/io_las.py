#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import laspy


@dataclass
class LasCloud:
    las: laspy.LasData
    xyz: np.ndarray
    labels: Optional[np.ndarray] = None


def read_las(path: str, label_field: str = "classification") -> LasCloud:
    las = laspy.read(path)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)

    labels = None
    if label_field:
        try:
            if hasattr(las, label_field):
                labels = np.asarray(getattr(las, label_field)).astype(np.uint8)
            elif label_field in las.point_format.dimension_names:
                labels = np.asarray(las[label_field]).astype(np.uint8)
        except Exception:
            labels = None

    return LasCloud(las=las, xyz=xyz, labels=labels)


def _ensure_extra_dim(las: laspy.LasData, name: str, dtype: np.dtype, description: str = "") -> None:
    existing = set(las.point_format.dimension_names)
    if name in existing:
        return

    dt = np.dtype(dtype)
    if dt == np.uint8:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint8, description=description))
    elif dt == np.float32:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.float32, description=description))
    else:
        raise ValueError(f"Unsupported extra dim dtype for {name}: {dt}")


def _clone_las_safe(src_las: laspy.LasData) -> laspy.LasData:
    """
    Safe clone for laspy objects without deepcopy.
    """
    las = laspy.LasData(src_las.header.copy())
    las.points = src_las.points.copy()

    try:
        las.vlrs = list(src_las.vlrs)
    except Exception:
        pass

    try:
        if getattr(src_las, "evlrs", None) is not None:
            las.evlrs = list(src_las.evlrs)
    except Exception:
        pass

    return las


def _clear_classification_flags_if_present(las: laspy.LasData, n_pts: int) -> None:
    """
    Reset class-related flags that may survive from the source LAS and confuse
    downstream viewers after Classification is overwritten.
    """
    zero_u8 = np.zeros(n_pts, dtype=np.uint8)

    for attr in [
        "synthetic",
        "key_point",
        "withheld",
        "overlap",
        "scanner_channel",
    ]:
        if hasattr(las, attr):
            try:
                setattr(las, attr, zero_u8)
            except Exception:
                pass


def write_las_with_classification(
    cloud_or_las: Union[LasCloud, laspy.LasData],
    out_path: str,
    pred_labels: np.ndarray,
    pred_leaf_prob: Optional[np.ndarray] = None,
    overwrite_classification: bool = True,
    save_pred_class_dim: bool = True,
    save_pred_prob_dim: bool = True,
) -> None:
    """
    Writes prediction to LAS/LAZ.

    Behavior:
    - overwrites Classification if requested
    - also stores pred_class as an extra dimension
    - optionally stores pred_leaf_prob as float32 extra dimension
    - preserves all existing input attributes
    """
    if isinstance(cloud_or_las, LasCloud):
        src_las = cloud_or_las.las
    else:
        src_las = cloud_or_las

    las = _clone_las_safe(src_las)

    pred_labels = np.asarray(pred_labels, dtype=np.uint8).reshape(-1)
    n_pts = len(las.points)

    if pred_labels.shape[0] != n_pts:
        raise ValueError(f"pred_labels length {pred_labels.shape[0]} != number of points {n_pts}")

    if pred_leaf_prob is not None:
        pred_leaf_prob = np.asarray(pred_leaf_prob, dtype=np.float32).reshape(-1)
        if pred_leaf_prob.shape[0] != n_pts:
            raise ValueError(f"pred_leaf_prob length {pred_leaf_prob.shape[0]} != number of points {n_pts}")

    if overwrite_classification:
        las.classification = pred_labels
        _clear_classification_flags_if_present(las, n_pts)

    if save_pred_class_dim:
        _ensure_extra_dim(las, "pred_class", np.uint8, "Model-predicted class")
        las["pred_class"] = pred_labels

    if save_pred_prob_dim and pred_leaf_prob is not None:
        _ensure_extra_dim(las, "pred_leaf_prob", np.float32, "Model probability for leaf class")
        las["pred_leaf_prob"] = pred_leaf_prob.astype(np.float32)

    las.write(out_path)