from __future__ import annotations

"""Configuration validation for the original points2SBL pipeline.

This module intentionally validates only the mature legacy pipeline used by
``prepare_data.py``, ``train.py`` and ``predict.py``.  It does not alter their
runtime behavior; it only validates the YAML schema and derives the feature
width shown by ``points2sbl validate-config``.

The repository has historically used a mapping-style ``features`` section,
for example::

    features:
      use_xyz: true
      use_centered_xyz: true
      include_geom_features: true
      geom_select: [linearity, planarity, scattering, curvature]

A small list-style compatibility path is retained for older external configs.
"""

import copy
import json
from typing import Any, Dict, List, Mapping, Sequence


class ConfigError(ValueError):
    """Raised when a YAML configuration is invalid."""


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = d
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _require(d: Dict[str, Any], path: str) -> Any:
    value = _get(d, path, None)
    if value is None:
        raise ConfigError(f"Missing required config field: {path!r}")
    return value


def _validate_model(cfg: Dict[str, Any]) -> None:
    model = _require(cfg, "model")
    if not isinstance(model, dict):
        raise ConfigError("'model' must be a mapping.")

    name = _require(cfg, "model.name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("'model.name' must be a non-empty string.")

    supported = {"point_transformer", "pointnet2", "pointnext"}
    normalized_name = name.strip().lower()
    if normalized_name not in supported:
        raise ConfigError(
            f"Unsupported model.name={name!r}. Supported models: "
            "point_transformer, pointnet2, pointnext."
        )

    num_classes = _require(cfg, "model.num_classes")
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ConfigError("'model.num_classes' must be an integer >= 2.")

    configured_in_dim = model.get("in_dim")
    if configured_in_dim is not None:
        try:
            configured_in_dim = int(configured_in_dim)
        except (TypeError, ValueError) as exc:
            raise ConfigError("'model.in_dim' must be an integer when set.") from exc
        if configured_in_dim < 1:
            raise ConfigError("'model.in_dim' must be >= 1 when set.")


def _validate_feature_mapping(features: Mapping[str, Any]) -> None:
    boolean_keys = (
        "use_xyz",
        "use_centered_xyz",
        "include_geom_features",
        "use_normals",
        "use_eigen",
        "normalize_features",
        "standardize_features",
        "log_omnivariance",
    )
    for key in boolean_keys:
        if key in features and not isinstance(features[key], bool):
            raise ConfigError(f"'features.{key}' must be boolean.")

    geom_select = features.get("geom_select")
    if geom_select is not None:
        if not isinstance(geom_select, (list, tuple)) or any(
            not isinstance(value, str) for value in geom_select
        ):
            raise ConfigError("'features.geom_select' must be a list of strings.")

        allowed = {
            "linearity",
            "planarity",
            "scattering",
            "omnivariance",
            "anisotropy",
            "eigenentropy",
            "curvature",
        }
        unknown = [
            value for value in geom_select
            if value.strip().lower() not in allowed
        ]
        if unknown:
            raise ConfigError(
                "Unknown features.geom_select values: "
                + ", ".join(repr(value) for value in unknown)
            )

    geom_k = features.get("geom_k")
    if geom_k is not None:
        values = geom_k if isinstance(geom_k, (list, tuple)) else [geom_k]
        try:
            parsed = [int(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ConfigError("'features.geom_k' must contain integers.") from exc
        if any(value < 3 for value in parsed):
            raise ConfigError("All 'features.geom_k' values must be >= 3.")

    for key in ("geom_k_infer", "normal_k", "eigen_k"):
        if key in features:
            try:
                value = int(features[key])
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"'features.{key}' must be an integer.") from exc
            if value < 3:
                raise ConfigError(f"'features.{key}' must be >= 3.")

    if "clip_features" in features:
        try:
            clip = float(features["clip_features"])
        except (TypeError, ValueError) as exc:
            raise ConfigError("'features.clip_features' must be numeric.") from exc
        if clip < 0.0:
            raise ConfigError("'features.clip_features' must be >= 0.")


def _compute_mapping_in_dim(features: Mapping[str, Any]) -> int:
    """Mirror the mature dataset/train feature-width behavior.

    ``use_centered_xyz`` selects centered XYZ *instead of* raw XYZ whenever
    ``use_xyz`` is enabled; it therefore does not add another three channels.
    This is how ``dataset.py`` and ``train.py`` currently construct features.
    """
    in_dim = 0

    use_xyz = bool(features.get("use_xyz", True))
    if use_xyz:
        in_dim += 3

    include_geom = bool(features.get("include_geom_features", False))
    if include_geom:
        selected = features.get(
            "geom_select",
            ["linearity", "planarity", "scattering"],
        )
        if not isinstance(selected, (list, tuple)) or len(selected) == 0:
            selected = ["linearity", "planarity", "scattering"]
        in_dim += len(selected)
    else:
        # Compatibility with the older full-geometry feature toggles.
        if bool(features.get("use_normals", False)):
            in_dim += 3
        if bool(features.get("use_eigen", False)):
            in_dim += 7

    # The mature data path falls back to a one-channel zero feature tensor only
    # when every feature family is disabled. Preserve a useful derived width.
    return max(int(in_dim), 1)


def _compute_list_in_dim(features: Sequence[str]) -> int:
    """Compatibility path for early list-style external configurations."""
    in_dim = 3
    for feature in features:
        if not isinstance(feature, str):
            raise ConfigError("Every entry in 'features' must be a string.")

        name = feature.lower().strip()
        if name in ("xyz", "xyz_centered"):
            continue
        if name in ("height", "z", "rel_z", "intensity", "curvature"):
            in_dim += 1
        elif name == "rgb":
            in_dim += 3
        elif name in ("normals", "normal"):
            in_dim += 3
        else:
            raise ConfigError(f"Unknown legacy feature {feature!r}.")
    return int(in_dim)


def compute_in_dim(cfg: Dict[str, Any]) -> int:
    """Compute the feature width implied by the original pipeline config."""
    features = _get(cfg, "features", {})
    if features is None:
        features = {}

    if isinstance(features, Mapping):
        _validate_feature_mapping(features)
        return _compute_mapping_in_dim(features)

    if isinstance(features, list):
        return _compute_list_in_dim(features)

    raise ConfigError("'features' must be a mapping (recommended) or a list.")


def _validate_train(cfg: Dict[str, Any]) -> None:
    train = _require(cfg, "train")
    if not isinstance(train, dict):
        raise ConfigError("'train' must be a mapping.")

    epochs = _require(cfg, "train.epochs")
    if not isinstance(epochs, int) or epochs < 1:
        raise ConfigError("'train.epochs' must be an integer >= 1.")

    learning_rate = _require(cfg, "train.lr")
    try:
        learning_rate = float(learning_rate)
    except (TypeError, ValueError) as exc:
        raise ConfigError("'train.lr' must be numeric.") from exc
    if learning_rate <= 0.0:
        raise ConfigError("'train.lr' must be > 0.")


def _validate_data(cfg: Dict[str, Any]) -> None:
    data = _require(cfg, "data")
    if not isinstance(data, dict):
        raise ConfigError("'data' must be a mapping.")

    for key in ("train_dir", "val_dir"):
        value = _require(cfg, f"data.{key}")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"'data.{key}' must be a non-empty string.")


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate a configuration for the mature points2SBL pipeline."""
    if not isinstance(cfg, dict):
        raise ConfigError("Config must be a mapping.")

    _validate_model(cfg)
    _validate_train(cfg)
    _validate_data(cfg)

    derived_in_dim = compute_in_dim(cfg)
    configured_in_dim = _get(cfg, "model.in_dim", None)
    if configured_in_dim is not None and int(configured_in_dim) != derived_in_dim:
        raise ConfigError(
            f"model.in_dim={configured_in_dim}, but the configured feature "
            f"pipeline produces {derived_in_dim} channels."
        )


def resolve_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and return a deep-copied config with derived runtime fields."""
    validate_config(cfg)
    output = copy.deepcopy(cfg)
    output.setdefault("runtime", {})
    output["runtime"]["in_dim"] = compute_in_dim(output)
    output["runtime"]["pipeline_version"] = 1
    return output


def dump_resolved_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(cfg, stream, indent=2, sort_keys=True)


__all__ = [
    "ConfigError",
    "compute_in_dim",
    "dump_resolved_config",
    "resolve_config",
    "validate_config",
]
