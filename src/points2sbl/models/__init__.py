from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .point_transformer import (
    PointTransformerConfig,
    PointTransformerSeg,
)
from .pointnet2 import (
    PointNet2Config,
    PointNet2Seg,
)
from .pointnext import (
    PointNeXtConfig,
    PointNeXtSeg,
)


def _filter_kwargs_for_dataclass(
    cls,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", None)

    if not fields:
        return kwargs

    allowed = set(fields)

    return {
        key: value
        for key, value in kwargs.items()
        if key in allowed
    }


def build_model(
    cfg_or_model_cfg: Dict[str, Any],
    in_dim: int,
) -> nn.Module:
    """
    Build one of the supported points2SBL segmentation models.

    Supported models
    ----------------
    point_transformer
    pointnet2
    pointnext

    Parameters
    ----------
    cfg_or_model_cfg
        Either the complete resolved configuration containing a ``model``
        section, or the model configuration dictionary itself.

    in_dim
        Number of input feature channels.

    Returns
    -------
    nn.Module
        Initialized segmentation model.
    """

    if (
        "model" in cfg_or_model_cfg
        and isinstance(
            cfg_or_model_cfg["model"],
            dict,
        )
    ):
        model_cfg = cfg_or_model_cfg["model"]
    else:
        model_cfg = cfg_or_model_cfg

    name = str(
        model_cfg.get(
            "name",
            "point_transformer",
        )
    ).lower()

    num_classes = int(
        model_cfg.get(
            "num_classes",
            2,
        )
    )

    dropout = float(
        model_cfg.get(
            "dropout",
            model_cfg.get(
                "pn2_dropout",
                0.2,
            ),
        )
    )

    feat_dropout = float(
        model_cfg.get(
            "feat_dropout",
            0.0,
        )
    )

    # -------------------------------------------------------------------------
    # PointNet++
    # -------------------------------------------------------------------------

    if name == "pointnet2":
        cfg = PointNet2Config(
            num_classes=num_classes,
            input_dim=int(in_dim),
            sa1_npoint=int(
                model_cfg.get(
                    "sa1_npoint",
                    2048,
                )
            ),
            sa1_k=int(
                model_cfg.get(
                    "sa1_k",
                    32,
                )
            ),
            sa2_npoint=int(
                model_cfg.get(
                    "sa2_npoint",
                    512,
                )
            ),
            sa2_k=int(
                model_cfg.get(
                    "sa2_k",
                    32,
                )
            ),
            sa3_npoint=int(
                model_cfg.get(
                    "sa3_npoint",
                    128,
                )
            ),
            sa3_k=int(
                model_cfg.get(
                    "sa3_k",
                    32,
                )
            ),
            dropout=dropout,
            feat_dropout=feat_dropout,
        )

        return PointNet2Seg(cfg)

    # -------------------------------------------------------------------------
    # PointNeXt
    # -------------------------------------------------------------------------

    if name == "pointnext":
        kwargs = dict(
            num_classes=num_classes,
            in_dim=int(in_dim),
            input_dim=int(in_dim),

            sa1_npoint=int(
                model_cfg.get(
                    "sa1_npoint",
                    2048,
                )
            ),
            sa1_k=int(
                model_cfg.get(
                    "sa1_k",
                    32,
                )
            ),

            sa2_npoint=int(
                model_cfg.get(
                    "sa2_npoint",
                    512,
                )
            ),
            sa2_k=int(
                model_cfg.get(
                    "sa2_k",
                    32,
                )
            ),

            sa3_npoint=int(
                model_cfg.get(
                    "sa3_npoint",
                    128,
                )
            ),
            sa3_k=int(
                model_cfg.get(
                    "sa3_k",
                    32,
                )
            ),

            width1=int(
                model_cfg.get(
                    "width1",
                    64,
                )
            ),
            width2=int(
                model_cfg.get(
                    "width2",
                    128,
                )
            ),
            width3=int(
                model_cfg.get(
                    "width3",
                    256,
                )
            ),

            dropout=dropout,
            feat_dropout=feat_dropout,

            blocks_per_stage=int(
                model_cfg.get(
                    "blocks_per_stage",
                    2,
                )
            ),

            expand_ratio=int(
                model_cfg.get(
                    "expand_ratio",
                    4,
                )
            ),

            act=str(
                model_cfg.get(
                    "act",
                    "relu",
                )
            ),

            drop_path=float(
                model_cfg.get(
                    "drop_path",
                    0.0,
                )
            ),

            use_dist=bool(
                model_cfg.get(
                    "use_dist",
                    True,
                )
            ),

            use_pre_mlp=bool(
                model_cfg.get(
                    "use_pre_mlp",
                    True,
                )
            ),
        )

        cfg = PointNeXtConfig(
            **_filter_kwargs_for_dataclass(
                PointNeXtConfig,
                kwargs,
            )
        )

        return PointNeXtSeg(cfg)

    # -------------------------------------------------------------------------
    # Point Transformer
    # -------------------------------------------------------------------------

    if name == "point_transformer":
        cfg = PointTransformerConfig(
            num_classes=num_classes,
            in_dim=int(in_dim),

            dim=int(
                model_cfg.get(
                    "dim",
                    128,
                )
            ),

            depth=int(
                model_cfg.get(
                    "depth",
                    6,
                )
            ),

            knn_k=int(
                model_cfg.get(
                    "knn_k",
                    24,
                )
            ),

            dropout=float(
                model_cfg.get(
                    "dropout",
                    0.1,
                )
            ),
        )

        return PointTransformerSeg(cfg)

    # -------------------------------------------------------------------------
    # Unsupported model
    # -------------------------------------------------------------------------

    raise ValueError(
        f"Unknown model name: {name}. "
        "Supported models: "
        "point_transformer, pointnet2, pointnext"
    )


__all__ = [
    "PointTransformerSeg",
    "PointTransformerConfig",
    "PointNet2Seg",
    "PointNet2Config",
    "PointNeXtSeg",
    "PointNeXtConfig",
    "build_model",
]