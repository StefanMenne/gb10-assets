import importlib.util

import pytest
from pydantic import ValidationError

from gb10_assets.server.config import REPO_ROOT
from gb10_assets.shared.schemas import (
    BLENDER_SCRIPTS,
    REQUESTS,
    BlenderScript,
    EditImage,
    GenerateImage,
    GenerateVideo,
    ImageTo3D,
    JobState,
    Speak,
    SpeakMultilingual,
    check_file_name,
    ltx_frames,
    ltx_max_seconds,
    ltx_memory_gb,
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


@pytest.mark.parametrize("size, seconds, fits", [
    ((1536, 1024), 6, True), ((1536, 1024), 7, False), ((1536, 1024), 10, False),  # 10 s ran out of memory
    ((1280, 832), 9, True), ((1280, 832), 11, False), ((1024, 704), 13, True), ((768, 512), 20, True),
])
def test_video_memory_limit(size, seconds, fits):
    params = {"prompt": "x", "width": size[0], "height": size[1], "duration_s": seconds}
    if fits:
        GenerateVideo.model_validate(params)
    else:
        with pytest.raises(ValidationError, match="at most"):
            GenerateVideo.model_validate(params)


def test_video_memory_limit_needs_size_and_length():
    GenerateVideo(prompt="x", duration_s=20)  # default size: the worker shrinks it
    GenerateVideo(prompt="x", width=2048, height=2048)  # predicted length: the worker caps it
    with pytest.raises(ValidationError, match="at most"):  # several keyframes: 5 s
        GenerateVideo(prompt="x", images=["a.png", "b.png"], width=2048, height=2048)
    GenerateVideo(prompt="x", video="in.mp4", start_s=1, end_s=2)  # retake: checked by the worker


def test_video_memory_model_matches_the_tool():
    """The schema rejects early what tools/ltx-video/run.py would refuse; both must compute the same."""
    spec = importlib.util.spec_from_file_location("ltx_run", REPO_ROOT / "tools" / "ltx-video" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    from gb10_assets.shared.schemas import LTX_MAX_GB
    assert run.MAX_GB == LTX_MAX_GB
    for w, h in [(1536, 1024), (1280, 832), (768, 512), (2048, 2048)]:
        for fps in (12, 24, 50):
            assert run.max_frames(w, h, fps) == round(ltx_max_seconds(w, h, fps) * fps) + 1
            for seconds in (1, 5, 13.3, 20):
                frames = run.frames_for(seconds, fps)
                assert frames == ltx_frames(seconds, fps)
                assert run.memory_gb(w, h, frames, fps) == ltx_memory_gb(w, h, frames, fps)
    w, h = run.fit_size(1536, 1024, run.frames_for(20, 24), 24)
    assert run.memory_gb(w, h, run.frames_for(20, 24), 24) <= run.MAX_GB and w % 64 == h % 64 == 0
    assert abs(w / h - 1.5) < 0.1


def test_image_to_3d():
    r = ImageTo3D(image="chair.png")
    assert (r.remove_background, r.resolution, r.max_faces, r.texture_size, r.count) == (True, 1536, 700_000, 4096, 1)
    assert r.input_files() == ["chair.png"] and REQUESTS["pixal3d"] is ImageTo3D


@pytest.mark.parametrize("params", [
    {},
    {"image": "a.png", "resolution": 1280},
    {"image": "a.png", "max_faces": 5},
    {"image": "a.png", "texture_size": 3000},
    {"image": "a.png", "count": 5},
    {"image": "a.png", "prompt": "x"},
])
def test_image_to_3d_invalid(params):
    with pytest.raises(ValidationError):
        ImageTo3D.model_validate(params)


def test_blender_script():
    r = BlenderScript(script="rig_hop", files=["fox.glb"], args=["fox.glb", "fox_rig.glb"])
    assert r.input_files() == ["fox.glb"] and REQUESTS["blender"] is BlenderScript
    assert BlenderScript(code="import bpy").args == []


@pytest.mark.parametrize("params", [
    {},
    {"script": "rig_hop", "code": "print(1)"},
    {"script": "no_such_script"},
    {"code": ""},
    {"script": "rig_hop", "files": ["../x.glb"]},
    {"script": "rig_hop", "args": ["x" * 4001]},
])
def test_blender_script_invalid(params):
    with pytest.raises(ValidationError):
        BlenderScript.model_validate(params)


def test_blender_scripts_match_the_tool():
    """BLENDER_SCRIPTS lists exactly the runnable scripts in tools/blender/scripts (texlib is a module)."""
    scripts = {p.stem for p in (REPO_ROOT / "tools" / "blender" / "scripts").glob("*.py")} - {"texlib"}
    assert scripts == set(BLENDER_SCRIPTS)


@pytest.mark.parametrize("name", ["a/b.png", "..png", ".hidden", "", "x" * 200, "a b.png"])
def test_bad_file_names(name):
    with pytest.raises(ValueError):
        check_file_name(name)


def test_finished_states():
    assert JobState.done.finished and JobState.failed.finished and JobState.cancelled.finished
    assert not JobState.queued.finished and not JobState.running.finished
