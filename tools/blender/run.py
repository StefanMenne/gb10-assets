"""Blender worker: runs a Python script in Blender per job (bpy, Blender 5.2 as a Python module, built by
post_install.sh: Cycles on CUDA, EEVEE through EGL, no display).

A job is one of the bundled scripts in scripts/ (by name) or own code, with arguments as on a command line
(sys.argv[1:]). The script runs in the job's output directory, where the input files are linked in under their
names: it reads them by name and writes its results next to them under new names; every new file is a result
(not subdirectories; names of letters, digits, . _ -). print() goes to the job log. SystemExit(0) ends a script
normally, any other exit or exception fails the job; a job without results fails too, except one asking only
for help (-h/--help: argparse prints the usage into the log and exits). scripts/ is on sys.path, so own code can import texlib.
Blender is reset to factory settings (an empty scene) before every job; bpy stays loaded between jobs.
"""
import os
import re
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _worker import log, serve  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent / "scripts"
HELP = {"-h", "--help"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")  # the server's file name rule (shared/schemas.py)


def load():
    import bpy

    sys.path.insert(0, str(SCRIPTS))
    log(f"Blender {bpy.app.version_string}")
    return bpy


def run(bpy, params, job_dir):
    out = job_dir / "out"
    links = []
    for f in sorted((job_dir / "in").iterdir()):
        (out / f.name).symlink_to(f)
        links.append(out / f.name)
    name = params.get("script") or "code"
    argv, cwd = sys.argv, os.getcwd()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    os.chdir(out)
    sys.argv = [name, *params.get("args", [])]
    try:
        if params.get("script"):
            runpy.run_path(str(SCRIPTS / f"{name}.py"), run_name="__main__")
        else:
            exec(compile(params["code"], "<code>", "exec"), {"__name__": "__main__", "__file__": "<code>"})
    except SystemExit as e:
        if e.code not in (None, 0):
            raise RuntimeError(e.code if isinstance(e.code, str) else f"{name} exited with code {e.code}") from None
    finally:
        sys.argv = argv
        os.chdir(cwd)
        for link in links:
            if link.is_symlink():
                link.unlink()
    files = sorted(p.name for p in out.iterdir() if p.is_file() and not p.is_symlink())
    outputs = [f for f in files if SAFE_NAME.match(f)]
    if skipped := sorted(set(files) - set(outputs)) + [p.name + "/" for p in out.iterdir() if p.is_dir()]:
        log(f"not returned (subdirectory, or a name other than letters, digits, . _ -): {', '.join(skipped)}")
    if not outputs and not HELP & set(params.get("args", [])):
        raise RuntimeError(f"{name} wrote no files into its working directory")
    return outputs


if __name__ == "__main__":
    serve(load, run)
