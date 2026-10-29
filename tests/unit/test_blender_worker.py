"""tools/blender/run.py without Blender: a stand-in bpy module, the scripts run for real."""
import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest

from gb10_assets.server.config import REPO_ROOT


@pytest.fixture
def worker(monkeypatch):
    resets = []
    bpy = SimpleNamespace(ops=SimpleNamespace(wm=SimpleNamespace(
        read_factory_settings=lambda use_empty: resets.append(use_empty))))
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    spec = importlib.util.spec_from_file_location("blender_run", REPO_ROOT / "tools" / "blender" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.resets = resets
    return module


@pytest.fixture
def job_dir(tmp_path):
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "in" / "fox.txt").write_text("fox")
    return tmp_path


def run(worker, job_dir, **params):
    return worker.run(sys.modules["bpy"], {"args": [], **params}, job_dir)


def test_code_reads_inputs_and_returns_new_files(worker, job_dir):
    code = ("import sys\n"
            "open(sys.argv[2], 'w').write(open(sys.argv[1]).read().upper())\n"
            "open('bad name.txt', 'w').write('x')\n")
    argv, cwd = list(sys.argv), os.getcwd()
    assert run(worker, job_dir, code=code, args=["fox.txt", "FOX.txt"]) == ["FOX.txt"]
    assert (job_dir / "out" / "FOX.txt").read_text() == "FOX"
    assert not (job_dir / "out" / "fox.txt").exists()          # the input link is gone again
    assert (job_dir / "in" / "fox.txt").read_text() == "fox"
    assert (sys.argv, os.getcwd(), worker.resets) == (argv, cwd, [True])


def test_bundled_script_by_name(worker, job_dir, tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "copy.py").write_text("import shutil, sys\nshutil.copy(sys.argv[1], sys.argv[2])\nraise SystemExit(0)\n")
    monkeypatch.setattr(worker, "SCRIPTS", scripts)
    assert run(worker, job_dir, script="copy", args=["fox.txt", "fox2.txt"]) == ["fox2.txt"]


def test_help_returns_no_files_without_failing(worker, job_dir, capsys):
    code = ("import argparse\n"
            "argparse.ArgumentParser(prog='tex_eyes').add_argument('--eyes', help='eye centres by hand')\n"
            "argparse.ArgumentParser(prog='tex_eyes').parse_args()\n")
    assert run(worker, job_dir, code=code, args=["--help"]) == []
    assert "usage: tex_eyes" in capsys.readouterr().out


@pytest.mark.parametrize("code, message", [
    ("raise SystemExit(2)", "code exited with code 2"),
    ("raise SystemExit('no pair of eyes found')", "no pair of eyes found"),
    ("x = 1", "code wrote no files"),
    ("raise ValueError('boom')", "boom"),
])
def test_failures(worker, job_dir, code, message):
    with pytest.raises(Exception, match=message):
        run(worker, job_dir, code=code)
    assert not (job_dir / "out" / "fox.txt").exists()
