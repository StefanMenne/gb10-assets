"""Worker harness shared by all tools/<tool>/run.py (standard library only, runs in the tool's own environment).

The server starts ``python tools/<tool>/run.py`` and talks JSON lines over stdin/stdout:

    server -> worker   {"job_dir": "/jobs/<id>", "params": {...}}       one line per job
    worker -> server   {"event": "ready"}                                 model loaded
                       {"event": "done", "outputs": ["a.png", ...]}       names inside <job_dir>/out
                       {"event": "error", "message": "..."}               job failed, worker keeps running
                       {"event": "fatal", "message": "..."}               loading failed, worker exits

Everything else (library output, progress, tracebacks) goes to stderr, which the server writes into the
job log; before each event a SYNC line on stderr tells the server that the log up to the event has arrived. File descriptor 1 is redirected to stderr so stray prints cannot corrupt the protocol. The worker
loads the model once and serves jobs until stdin closes (sticky worker), then exits and frees GPU memory.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path


SYNC = "\x00sync"


def log(msg):
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def _protocol_stream():
    fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(fd, "w", buffering=1)


def serve(load, run):
    """load() -> model; run(model, params: dict, job_dir: Path) -> list of output file names in job_dir/out."""
    out = _protocol_stream()

    def send(**msg):
        print(SYNC, file=sys.stderr, flush=True)
        out.write(json.dumps(msg) + "\n")
        out.flush()

    t = time.time()
    try:
        model = load()
    except BaseException as e:  # noqa: BLE001 - report every load failure to the server
        traceback.print_exc()
        send(event="fatal", message=f"loading failed: {e!r}")
        sys.exit(1)
    log(f"model loaded in {time.time() - t:.0f}s")
    send(event="ready")

    for line in sys.stdin:
        if not line.strip():
            continue
        job = json.loads(line)
        job_dir = Path(job["job_dir"])
        (job_dir / "out").mkdir(parents=True, exist_ok=True)
        t = time.time()
        try:
            outputs = run(model, job["params"], job_dir)
        except Exception as e:  # noqa: BLE001 - job errors go to the client, the worker stays up
            traceback.print_exc()
            send(event="error", message=repr(e))
            continue
        log(f"job done in {time.time() - t:.0f}s: {', '.join(outputs)}")
        send(event="done", outputs=outputs)


def seeds(params):
    """Seeds seed..seed+count-1; a random start seed when none is given."""
    start = params.get("seed")
    if start is None:
        start = int.from_bytes(os.urandom(4), "little") % (2**31)
    return [start + k for k in range(params.get("count", 1))]


def contact_sheet(paths, out, size=400):
    """Variants side by side with their file names (needs Pillow in the tool environment)."""
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (size * len(paths), size + 30), "white")
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im.thumbnail((size, size))
        sheet.paste(im, (size * i, 30))
        draw.text((size * i + 8, 8), Path(p).name, fill="black")
    sheet.save(out)
    return Path(out).name
