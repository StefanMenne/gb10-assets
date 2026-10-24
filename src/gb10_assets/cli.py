"""gb10-assets command line.

    gb10-assets mcp                                   MCP server over stdio (for Claude Code)
    gb10-assets server [--host 0.0.0.0] [--port 8750] HTTP server on the GB10 system
    gb10-assets health                                server version and queue
    gb10-assets image "prompt" -o DIR [-n 3] [--seed 1] [-W 1024 -H 1024] [-i ref.png]
    gb10-assets edit "instruction" -i in.png [-i in2.png] -o DIR [-n 2] [--seed 1] [--steps 40] [--cfg 4]
    gb10-assets status JOB | fetch JOB DIR [--wait 600] | log JOB | cancel JOB

Generating commands wait for the result by default (--no-wait: print the job id and return).
"""
from __future__ import annotations

import argparse
import sys


def _client():
    from .client.http_client import Client

    c = Client()
    c.check_version()
    return c


def _submit_and_fetch(args, tool: str, params: dict) -> None:
    c = _client()
    info = c.submit(tool, params)
    print(f"job {info.id} {info.state.value}" + (f" (position {info.queue_position})" if info.queue_position else ""),
          file=sys.stderr)
    if args.no_wait:
        print(info.id)
        return
    _, paths = c.fetch(info.id, args.out, wait=args.timeout)
    if not paths:
        sys.exit(f"job {info.id} not finished after {args.timeout}s; later: gb10-assets fetch {info.id} {args.out}")
    for p in paths:
        print(p)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="gb10-assets", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mcp", help="MCP server over stdio")
    s = sub.add_parser("server", help="HTTP server (GB10 system)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8750)
    sub.add_parser("health", help="server version and queue state")

    gen = argparse.ArgumentParser(add_help=False)
    gen.add_argument("-o", "--out", default=".", help="target directory")
    gen.add_argument("--no-wait", action="store_true", help="only submit, print the job id")
    gen.add_argument("--timeout", type=float, default=3600, help="seconds to wait for the result")

    im = sub.add_parser("image", parents=[gen], help="text to image (FLUX.2 [klein])")
    im.add_argument("prompt")
    im.add_argument("-n", "--count", type=int, default=1)
    im.add_argument("--seed", type=int)
    im.add_argument("-W", "--width", type=int, default=1024)
    im.add_argument("-H", "--height", type=int, default=1024)
    im.add_argument("-i", "--image", action="append", default=[], help="reference image (repeatable)")

    ed = sub.add_parser("edit", parents=[gen], help="edit images with an instruction (Qwen-Image-Edit-2511)")
    ed.add_argument("prompt")
    ed.add_argument("-i", "--image", action="append", required=True, help="input image (1-3, repeatable)")
    ed.add_argument("-n", "--count", type=int, default=1)
    ed.add_argument("--seed", type=int)
    ed.add_argument("--steps", type=int, default=40)
    ed.add_argument("--cfg", type=float, default=4.0, help="true CFG scale")
    ed.add_argument("--negative", default="", help="negative prompt")
    ed.add_argument("-W", "--width", type=int, help="default: like the first image, about 1 megapixel")
    ed.add_argument("-H", "--height", type=int)

    for name in ("status", "log", "cancel"):
        sub.add_parser(name).add_argument("job")
    f = sub.add_parser("fetch", help="download results")
    f.add_argument("job")
    f.add_argument("out")
    f.add_argument("--wait", type=float, default=0)
    a = p.parse_args(argv)

    try:
        from .client.http_client import ApiError
    except ImportError:  # server-only install without the [client] extra
        class ApiError(Exception):  # type: ignore[no-redef]
            pass

    try:
        if a.cmd == "mcp":
            from .client.mcp_server import run
            run()
        elif a.cmd == "server":
            from .server.app import serve
            serve(a.host, a.port)
        elif a.cmd == "health":
            from .client.http_client import Client
            print(Client().health().model_dump_json(indent=2))
        elif a.cmd == "image":
            _submit_and_fetch(a, "flux2-klein", {"prompt": a.prompt, "count": a.count, "seed": a.seed,
                                                 "width": a.width, "height": a.height, "reference_images": a.image})
        elif a.cmd == "edit":
            _submit_and_fetch(a, "qwen-image-edit", {"prompt": a.prompt, "images": a.image, "count": a.count,
                                                     "seed": a.seed, "steps": a.steps, "true_cfg_scale": a.cfg,
                                                     "negative_prompt": a.negative, "width": a.width,
                                                     "height": a.height})
        elif a.cmd == "status":
            print(_client().status(a.job).model_dump_json(indent=2, exclude_none=True))
        elif a.cmd == "log":
            print(_client().log(a.job), end="")
        elif a.cmd == "cancel":
            _client().delete(a.job)
        elif a.cmd == "fetch":
            info, paths = _client().fetch(a.job, a.out, wait=a.wait)
            if not paths:
                sys.exit(f"job {a.job} is {info.state.value}")
            for path in paths:
                print(path)
    except ApiError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
