from points2sbl.model_manager import (
    MODEL_FILENAME,
    MODEL_SHA256,
    default_model_path,
)


def test_model_metadata_shape():
    assert MODEL_FILENAME.endswith(".pt")
    assert len(MODEL_SHA256) == 64
    int(MODEL_SHA256, 16)


def test_default_model_path_name():
    assert default_model_path().name == MODEL_FILENAME
