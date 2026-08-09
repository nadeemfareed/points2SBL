from __future__ import annotations

from pathlib import Path

import yaml

from points2sbl.config import resolve_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = REPO_ROOT / "configs" / name
    with path.open("r", encoding="utf-8-sig") as stream:
        return yaml.safe_load(stream)


def test_point_transformer_config_contract():
    resolved = resolve_config(_load("point_transformer.yaml"))
    assert resolved["runtime"]["in_dim"] == 7
    assert resolved["model"]["in_dim"] == 7
    assert resolved["model"]["name"] == "point_transformer"


def test_pointnet2_config_contract():
    resolved = resolve_config(_load("pointnet2.yaml"))
    assert resolved["runtime"]["in_dim"] == 7
    assert resolved["model"]["in_dim"] == 7
    assert resolved["model"]["name"] == "pointnet2"


def test_pointnext_config_contract():
    resolved = resolve_config(_load("pointnext.yaml"))
    assert resolved["runtime"]["in_dim"] == 7
    assert resolved["model"]["in_dim"] == 7
    assert resolved["model"]["name"] == "pointnext"
