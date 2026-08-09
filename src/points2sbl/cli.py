from __future__ import annotations

import argparse
import sys

from .config import resolve_config, ConfigError
from .utils import load_yaml


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Special-case predict help so predict.py prints its own full options
    if len(argv) >= 2 and argv[0] == "predict" and argv[1] in {"-h", "--help"}:
        from . import predict as predict_mod
        return int(predict_mod.main(["--help"]))

    p = argparse.ArgumentParser(
        prog="points2sbl",
        description="Deep learning segmentation of woody and leaf points in forest point clouds.",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser(
        "validate-config",
        help="Validate YAML configuration and show derived fields.",
    )
    p_val.add_argument(
        "--config",
        required=True,
        help="Path to YAML config.",
    )

    sub.add_parser(
        "predict",
        add_help=False,
        help="Run woody/leaf inference on LAS/LAZ files or folders.",
    )

    p_model = sub.add_parser(
        "model",
        help="Install or inspect the released pretrained Point Transformer.",
    )
    model_sub = p_model.add_subparsers(dest="model_cmd", required=True)

    p_model_download = model_sub.add_parser(
        "download",
        help="Download and SHA256-verify the released pretrained checkpoint.",
    )
    p_model_download.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing cached model.",
    )

    model_sub.add_parser(
        "status",
        help="Check whether the released checkpoint is installed and verified.",
    )
    model_sub.add_parser(
        "path",
        help="Print the default model-cache path.",
    )

    args, rest = p.parse_known_args(argv)

    if args.cmd == "validate-config":
        cfg = load_yaml(args.config)
        try:
            resolved = resolve_config(cfg)
        except ConfigError as e:
            print(f"[CONFIG ERROR] {e}", file=sys.stderr)
            return 2

        print("Config OK.")
        print(f"  model.name        : {resolved['model']['name']}")
        print(f"  model.num_classes : {resolved['model']['num_classes']}")
        print(f"  runtime.in_dim    : {resolved['runtime']['in_dim']}")
        print(f"  features          : {resolved.get('features', [])}")
        return 0

    if args.cmd == "predict":
        from . import predict as predict_mod
        return int(predict_mod.main(rest))

    if args.cmd == "model":
        from .model_manager import (
            default_model_path,
            download_default_model,
            model_status,
        )

        if args.model_cmd == "download":
            download_default_model(force=bool(args.force))
            return 0

        if args.model_cmd == "status":
            ok, message = model_status()
            print(message)
            return 0 if ok else 1

        if args.model_cmd == "path":
            print(default_model_path())
            return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())