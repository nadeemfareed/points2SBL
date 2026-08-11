from points2sbl.model_manager import (
    MODEL_ASSET_FILENAME,
    MODEL_LOCAL_FILENAME,
    MODEL_URL,
    default_model_path,
)


def test_model_asset_filename():
    assert MODEL_ASSET_FILENAME == "point_transformer_best.pt"


def test_default_model_path_name():
    assert default_model_path().name == MODEL_LOCAL_FILENAME


def test_model_url_uses_release_asset_name():
    assert MODEL_ASSET_FILENAME in MODEL_URL
