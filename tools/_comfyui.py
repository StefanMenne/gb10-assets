"""Headless ComfyUI for the workers that run ComfyUI workflows (standard library only).

ComfyUI (GPL-3.0) is fetched by the tool's post_install.sh into <install>/build/ComfyUI and runs as a separate
process on a local port; the worker sends it workflows in API format over its HTTP API and collects the files.
Models stay loaded in ComfyUI between jobs and are freed when the worker exits (ComfyUI exits with it).
"""
import atexit
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

READY_S = 300


class ComfyUI:
    def __init__(self, comfyui: Path, weights: Path, folders):
        """comfyui: the ComfyUI checkout; weights: a directory in ComfyUI's model folder layout with these folders."""
        self.base = Path(tempfile.mkdtemp(prefix="gb10-comfyui-"))
        (self.base / "frontend").mkdir()   # headless: no web frontend installed
        (self.base / "custom_nodes").mkdir()   # stays empty, ComfyUI only lists it
        paths = self.base / "extra_model_paths.yaml"
        paths.write_text("gb10:\n  base_path: " + json.dumps(str(weights)) + "\n"
                         + "".join(f"  {folder}: {folder}/\n" for folder in folders))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        # stdout too goes to stderr (the job log); same process group, so killing the worker kills ComfyUI
        self.proc = subprocess.Popen(
            [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(self.port), "--disable-auto-launch",
             "--base-directory", str(self.base), "--extra-model-paths-config", str(paths),
             "--front-end-root", str(self.base / "frontend")],
            cwd=comfyui, stdout=sys.stderr, stderr=sys.stderr)
        atexit.register(self.close)

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        shutil.rmtree(self.base, ignore_errors=True)

    def call(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)

    def check_alive(self):
        if self.proc.poll() is not None:
            # leave the worker: the server starts a new one (and ComfyUI with it) for the next job
            raise SystemExit(f"ComfyUI exited with code {self.proc.returncode}")

    def wait_ready(self):
        deadline = time.time() + READY_S
        while True:
            self.check_alive()
            try:
                return self.call("/system_stats")
            except (urllib.error.URLError, ConnectionError):
                if time.time() > deadline:
                    raise RuntimeError(f"ComfyUI not ready after {READY_S}s") from None
                time.sleep(1)

    def run(self, prompt, poll_s=1.0):
        """Queue a workflow in API format and wait for it; returns its outputs per node id."""
        try:
            pid = self.call("/prompt", {"prompt": prompt})["prompt_id"]
        except urllib.error.HTTPError as e:   # validation errors come as 400 with details
            raise RuntimeError(f"ComfyUI rejected the workflow: {e.read().decode()[:2000]}") from None
        while True:
            self.check_alive()
            entry = self.call(f"/history/{pid}").get(pid)
            if entry and entry["status"].get("completed") is not None:
                break
            time.sleep(poll_s)
        if entry["status"].get("status_str") != "success":
            errors = [m[1] for m in entry["status"].get("messages", []) if m[0] == "execution_error"]
            detail = "; ".join(f"{e.get('node_type')}: {e.get('exception_message', '').strip()}" for e in errors)
            raise RuntimeError(f"workflow failed: {detail or entry['status']}")
        return entry["outputs"]

    def output_file(self, item):
        return self.base / "output" / item.get("subfolder", "") / item["filename"]

    @property
    def input_dir(self):
        d = self.base / "input"
        d.mkdir(exist_ok=True)
        return d


FRONTEND_ONLY = {"Note", "MarkdownNote"}   # nodes of the editor that ComfyUI's server does not run
CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}   # the editor's "control after generate"


def api_format(ui_workflow, outputs):
    """A workflow saved by ComfyUI's editor (nodes and links) in API format, as the editor's "Export (API)" does:
    widget values by input name, links as [node id, output slot]. Only the given output nodes and the nodes they
    need are kept. Supports flat workflows: no subgraphs, no muted or bypassed nodes."""
    if ui_workflow.get("definitions", {}).get("subgraphs"):
        raise ValueError("workflows with subgraphs are not supported")
    links = {link[0]: (str(link[1]), link[2]) for link in ui_workflow["links"]}
    api = {}
    for node in ui_workflow["nodes"]:
        if node["type"] in FRONTEND_ONLY:
            continue
        if node.get("mode", 0) != 0:
            raise ValueError(f"node {node['id']} ({node['type']}) is muted or bypassed")
        widgets = [i for i in node.get("inputs", []) if "widget" in i]
        values = list(node.get("widgets_values") or [])
        inputs, k = {}, 0
        for n, widget in enumerate(widgets):
            if k >= len(values):
                raise ValueError(f"node {node['id']} ({node['type']}): no value for {widget['name']}")
            inputs[widget["name"]] = values[k]
            k += 1
            # an INT widget can be followed by its "control after generate" mode, which is no input
            if (widget["type"] == "INT" and k < len(values) and values[k] in CONTROL_VALUES
                    and len(values) - k > len(widgets) - n - 1):
                k += 1
        if k != len(values):
            raise ValueError(f"node {node['id']} ({node['type']}): {len(values)} widget values for "
                             f"{len(widgets)} widgets")
        for i in node.get("inputs", []):
            if i.get("link") is not None:
                inputs[i["name"]] = list(links[i["link"]])
        api[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
    return prune(api, outputs)


def prune(api, outputs):
    """The nodes of a workflow in API format that the given output nodes need, themselves included."""
    keep, todo = set(), [str(o) for o in outputs]
    while todo:
        nid = todo.pop()
        if nid in keep:
            continue
        keep.add(nid)
        todo += [v[0] for v in api[nid]["inputs"].values() if isinstance(v, list)]
    return {nid: node for nid, node in api.items() if nid in keep}


def start(comfyui: Path, weights: Path, files: dict, tool: str):
    """Check the installation, start ComfyUI and wait until it answers. files: {folder: name or (names)}."""
    from _worker import log

    missing = [f"{folder}/{name}" for folder, names in files.items()
               for name in ([names] if isinstance(names, str) else names) if not (weights / folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} missing in {weights}, install the tool on the GB10 system: "
                                f"scripts/docker.sh install {tool}")
    if not (comfyui / "main.py").exists():
        raise FileNotFoundError(f"ComfyUI missing in {comfyui}: scripts/docker.sh install {tool}")
    comfy = ComfyUI(comfyui, weights, files)
    stats = comfy.wait_ready()
    log(f"ComfyUI {stats['system'].get('comfyui_version')} on port {comfy.port}")
    return comfy
