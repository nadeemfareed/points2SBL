from pathlib import Path

from points2sbl.predict import _bundled_point_transformer_config, _argv_has_dest


def test_bundled_config_exists():
    assert Path(_bundled_point_transformer_config()).is_file()


def test_cli_override_detection():
    argv = ["--votes", "12", "--no-smooth", "--vote-mode", "random"]
    assert _argv_has_dest(argv, "votes")
    assert _argv_has_dest(argv, "smooth")
    assert _argv_has_dest(argv, "vote_mode")
    assert not _argv_has_dest(argv, "tile_size_m")
