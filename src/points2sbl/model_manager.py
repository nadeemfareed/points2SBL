from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from tqdm import tqdm

MODEL_FILENAME = "point_transformer_best.pt"
MODEL_VERSION = "0.3.0"
MODEL_URL = (
    "https://github.com/nadeemfareed/points2sbl/releases/download/"
    f"v{MODEL_VERSION}/{MODEL_FILENAME}"
)
MODEL_SHA256 = "fd43c5f83463f00d189292b4d4034bec21f3147c453232c4fbf8336cfd2047f9"
MODEL_BYTES = 18383398



def repo_root() -> Path:
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / "pyproject.toml").exists():
            return p
    return cwd


def model_cache_dir() -> Path:
    return (
        repo_root()
        / "runs"
        / "point_transformer_curated_20260327_170108"
    )


def default_model_path() -> Path:
    return model_cache_dir() / "best.pt"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def model_status() -> tuple[bool, str]:
    path = default_model_path()
    if not path.is_file():
        return False, f"Model not installed: {path}"
    digest = sha256_file(path)
    if digest != MODEL_SHA256:
        return False, (
            f"Model exists but SHA256 does not match release v{MODEL_VERSION}: "
            f"{path}"
        )
    return True, f"Model ready: {path}"


def download_default_model(
    *,
    force: bool = False,
    url: Optional[str] = None,
) -> Path:
    target = default_model_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        digest = sha256_file(target)
        if digest == MODEL_SHA256:
            print(f"[MODEL] Already installed: {target}")
            return target
        raise RuntimeError(
            f"A model already exists at {target} but its SHA256 differs from "
            "the released model. Re-run with --force to replace it."
        )

    download_url = str(url or MODEL_URL)
    print(f"[MODEL] Downloading points2SBL Point Transformer v{MODEL_VERSION}")
    print(f"[MODEL] URL: {download_url}")
    print(f"[MODEL] Destination: {target}")

    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": "points2sbl-model-downloader/0.3.0"},
    )

    fd, temp_name = tempfile.mkstemp(
        prefix="points2sbl_model_",
        suffix=".part",
        dir=str(target.parent),
    )
    os.close(fd)
    temp = Path(temp_name)

    try:
        with urllib.request.urlopen(req) as response, temp.open("wb") as out:
            total = response.headers.get("Content-Length")
            total_n = int(total) if total and total.isdigit() else None
            with tqdm(
                total=total_n,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="model",
            ) as bar:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    bar.update(len(chunk))

        digest = sha256_file(temp)
        if digest != MODEL_SHA256:
            raise RuntimeError(
                "Downloaded checkpoint failed SHA256 verification. "
                f"Expected {MODEL_SHA256}, received {digest}."
            )

        shutil.move(str(temp), str(target))
        print(f"[MODEL] Installed: {target}")
        print(f"[MODEL] SHA256: {MODEL_SHA256}")
        return target
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass
