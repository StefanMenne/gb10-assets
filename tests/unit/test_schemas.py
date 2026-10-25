import pytest
from pydantic import ValidationError

from gb10_assets.shared.schemas import (
    REQUESTS,
    EditImage,
    GenerateImage,
    GenerateVideo,
    JobState,
    Speak,
    SpeakMultilingual,
    check_file_name,
)


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


def test_speak_design_and_clone():
    design = Speak(texts=["Hallo"], voice_description="tiefe, ruhige Stimme")
    assert (design.language, design.count, design.input_files()) == ("auto", 1, [])
    clone = Speak(texts=["a", "b"], reference_audio="ref.wav", reference_text="hi", language="de")
    assert clone.input_files() == ["ref.wav"]
    assert REQUESTS["qwen-tts"] is Speak


@pytest.mark.parametrize("params", [
    {"texts": ["x"]},  # no voice
    {"texts": ["x"], "voice_description": "v", "reference_audio": "a.wav"},  # both
    {"texts": ["x"], "voice_description": "v", "reference_text": "t"},
    {"texts": [], "voice_description": "v"},
    {"texts": [""], "voice_description": "v"},
    {"texts": ["x"] * 65, "voice_description": "v"},
    {"texts": ["x"], "voice_description": "v", "language": "th"},  # not a Qwen3-TTS language
    {"texts": ["x"], "voice_description": "v", "language": "German"},
])
def test_speak_invalid(params):
    with pytest.raises(ValidationError):
        Speak.model_validate(params)


def test_speak_multilingual():
    req = SpeakMultilingual(texts=["สวัสดีครับ"], language="th", reference_audio="de.wav")
    assert req.input_files() == ["de.wav"]
    assert REQUESTS["confucius-tts"] is SpeakMultilingual
    for params in ({"texts": ["x"], "language": "th"}, {"texts": ["x"], "reference_audio": "a.wav"},
                   {"texts": ["x"], "language": "auto", "reference_audio": "a.wav"}):
        with pytest.raises(ValidationError):
            SpeakMultilingual.model_validate(params)


def test_video_modes():
    t2v = GenerateVideo(prompt="a fox runs")
    assert (t2v.width, t2v.duration_s, t2v.fps, t2v.count, t2v.input_files()) == (None, None, 24, 1, [])
    keyframes = GenerateVideo(prompt="x", images=["a.png", "b.png"], width=1024, height=576, duration_s=6)
    assert keyframes.input_files() == ["a.png", "b.png"]
    assert GenerateVideo(prompt="x", audio="song.wav", images=["face.png"]).input_files() == ["face.png", "song.wav"]
    retake = GenerateVideo(prompt="x", video="in.mp4", start_s=1, end_s=2.5, regenerate="audio")
    assert retake.input_files() == ["in.mp4"]
    assert REQUESTS["ltx-video"] is GenerateVideo


@pytest.mark.parametrize("params", [
    {"prompt": ""},
    {"prompt": "x", "width": 1000, "height": 576},  # not a multiple of 64
    {"prompt": "x", "width": 1024},  # height missing
    {"prompt": "x", "duration_s": 30},
    {"prompt": "x", "count": 9},
    {"prompt": "x", "images": ["a.png"] * 5},
    {"prompt": "x", "audio": "a.wav", "images": ["a.png", "b.png"]},
    {"prompt": "x", "video": "in.mp4"},  # no time range
    {"prompt": "x", "video": "in.mp4", "start_s": 3, "end_s": 2},
    {"prompt": "x", "video": "in.mp4", "start_s": 1, "end_s": 2, "images": ["a.png"]},
    {"prompt": "x", "video": "in.mp4", "start_s": 1, "end_s": 2, "audio": "a.wav"},
    {"prompt": "x", "start_s": 1, "end_s": 2},  # time range without video
    {"prompt": "x", "regenerate": "video"},
])
def test_video_invalid(params):
    with pytest.raises(ValidationError):
        GenerateVideo.model_validate(params)


@pytest.mark.parametrize("name", ["a/b.png", "..png", ".hidden", "", "x" * 200, "a b.png"])
def test_bad_file_names(name):
    with pytest.raises(ValueError):
        check_file_name(name)


def test_finished_states():
    assert JobState.done.finished and JobState.failed.finished and JobState.cancelled.finished
    assert not JobState.queued.finished and not JobState.running.finished
