import pytest
from pydantic import ValidationError

from gb10_assets.shared.schemas import REQUESTS, EditImage, GenerateImage, JobState, check_file_name


def test_defaults_and_input_files():
    req = GenerateImage(prompt="a fox", reference_images=["a.png", "b.png"])
    assert (req.width, req.height, req.count, req.seed) == (1024, 1024, 1, None)
    assert req.input_files() == ["a.png", "b.png"]
    assert REQUESTS["flux2-klein"] is GenerateImage


@pytest.mark.parametrize("params", [
    {"prompt": ""},
    {"prompt": "x", "width": 1000},
    {"prompt": "x", "height": 4096},
    {"prompt": "x", "count": 0},
    {"prompt": "x", "count": 17},
    {"prompt": "x", "reference_images": ["../etc/passwd"]},
    {"prompt": "x", "unknown": 1},
])
def test_invalid(params):
    with pytest.raises(ValidationError):
        GenerateImage.model_validate(params)


def test_edit_defaults_and_input_files():
    req = EditImage(prompt="make it night", images=["a.png", "b.png"])
    assert (req.width, req.height, req.seed, req.count) == (None, None, None, 1)
    assert req.input_files() == ["a.png", "b.png"]
    assert REQUESTS["qwen-image-edit"] is EditImage


@pytest.mark.parametrize("params", [
    {"prompt": "x", "images": []},
    {"prompt": "x", "images": ["a.png", "b.png", "c.png", "d.png"]},
    {"prompt": "x"},
    {"prompt": "x", "images": ["a.png"], "width": 1000},
    {"prompt": "x", "images": ["a.png"], "steps": 40},
])
def test_edit_invalid(params):
    with pytest.raises(ValidationError):
        EditImage.model_validate(params)


@pytest.mark.parametrize("name", ["a/b.png", "..png", ".hidden", "", "x" * 200, "a b.png"])
def test_bad_file_names(name):
    with pytest.raises(ValueError):
        check_file_name(name)


def test_finished_states():
    assert JobState.done.finished and JobState.failed.finished and JobState.cancelled.finished
    assert not JobState.queued.finished and not JobState.running.finished
