"""gb10-assets command line.

    gb10-assets mcp                                   MCP server over stdio (for Claude Code)
    gb10-assets server [--host 0.0.0.0] [--port 8750] HTTP server on the GB10 system
    gb10-assets health                                server version and queue
    gb10-assets image "prompt" -o DIR [-n 3] [--seed 1] [-W 1024 -H 1024] [-i ref.png]
    gb10-assets edit "instruction" -i in.png [-i in2.png] -o DIR [-n 2] [--seed 1] [-W 1024 -H 768]
    gb10-assets voice "description" "text" -o DIR [-l de] [-n 4] [--seed 1]
    gb10-assets speak "text" ["text" ...] -r ref.wav [-t "transcript" | -T ref.txt] -o DIR [-l de]
    gb10-assets speak-multilingual "text" ["text" ...] -r ref.wav -l th -o DIR
    gb10-assets video "prompt" -o DIR [-i first.png [-i last.png]] [-d 5] [-W 1536 -H 1024] [--fps 24] [-n 2]
    gb10-assets audio-to-video "prompt" -a song.wav -o DIR [-i first.png] [-d 10]
    gb10-assets retake in.mp4 "prompt" --start 2 --end 4 -o DIR [--regenerate video|audio]
    gb10-assets 3d object.png -o DIR [--resolution 1024] [--faces 50000] [--texture 2048] [--keep-background]
    gb10-assets status JOB | fetch JOB DIR [--wait 600] | log JOB | cancel JOB

Generating commands wait for the result by default (--no-wait: print the job id and return).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
    ed.add_argument("-W", "--width", type=int, help="default: like the first image, about 1 megapixel")
    ed.add_argument("-H", "--height", type=int)

    vo = sub.add_parser("voice", parents=[gen], help="design a voice from a description (Qwen3-TTS VoiceDesign)")
    vo.add_argument("description")
    vo.add_argument("text", help="what the voice says")
    vo.add_argument("-l", "--language", default="auto")
    vo.add_argument("-n", "--count", type=int, default=4)
    vo.add_argument("--seed", type=int)

    sp = sub.add_parser("speak", parents=[gen], help="speak with a cloned voice (Qwen3-TTS Base)")
    sp.add_argument("texts", nargs="+")
    sp.add_argument("-r", "--ref", required=True, help="reference recording of the voice")
    ref_text = sp.add_mutually_exclusive_group()
    ref_text.add_argument("-t", "--ref-text", help="transcript of the reference")
    ref_text.add_argument("-T", "--ref-text-file", help="file with the transcript (e.g. the .txt of voice)")
    sp.add_argument("-l", "--language", default="auto")
    sp.add_argument("--seed", type=int)

    sm = sub.add_parser("speak-multilingual", parents=[gen],
                        help="speak with a cloned voice in other languages such as Thai (Confucius4-TTS)")
    sm.add_argument("texts", nargs="+")
    sm.add_argument("-r", "--ref", required=True, help="reference recording of the voice, any language")
    sm.add_argument("-l", "--language", required=True)
    sm.add_argument("--seed", type=int)

    vi = sub.add_parser("video", parents=[gen], help="text/images to video with sound (LTX-2.5)")
    vi.add_argument("prompt")
    vi.add_argument("-i", "--image", action="append", default=[],
                    help="keyframe image (repeatable, max 4): first frame, or first and last frame, ...")
    vi.add_argument("-d", "--duration", type=float, help="seconds (1-20); default predicted from the prompt")
    vi.add_argument("-W", "--width", type=int, help="multiple of 64; default 1536x1024 or the image's aspect")
    vi.add_argument("-H", "--height", type=int)
    vi.add_argument("--fps", type=int, default=24)
    vi.add_argument("--enhance", action="store_true", help="expand the prompt with the text encoder")
    vi.add_argument("-n", "--count", type=int, default=1)
    vi.add_argument("--seed", type=int)

    av = sub.add_parser("audio-to-video", parents=[gen], help="video for an audio track (LTX-2.5)")
    av.add_argument("prompt")
    av.add_argument("-a", "--audio", required=True, help="audio the video is generated for")
    av.add_argument("-i", "--image", help="first frame")
    av.add_argument("-d", "--duration", type=float, help="seconds; default the audio (max 20)")
    av.add_argument("-W", "--width", type=int)
    av.add_argument("-H", "--height", type=int)
    av.add_argument("-n", "--count", type=int, default=1)
    av.add_argument("--seed", type=int)

    rt = sub.add_parser("retake", parents=[gen], help="regenerate a time range of a video (LTX-2.5)")
    rt.add_argument("video")
    rt.add_argument("prompt")
    rt.add_argument("--start", type=float, required=True, help="seconds")
    rt.add_argument("--end", type=float, required=True, help="seconds")
    rt.add_argument("--regenerate", choices=["both", "video", "audio"], default="both")
    rt.add_argument("-n", "--count", type=int, default=1)
    rt.add_argument("--seed", type=int)

    d3 = sub.add_parser("3d", parents=[gen], help="image to textured 3D model, GLB (Pixal3D)")
    d3.add_argument("image")
    d3.add_argument("--resolution", type=int, choices=[1024, 1536], default=1536, help="1024: about twice as fast")
    d3.add_argument("--faces", type=int, default=700_000, help="faces of the mesh")
    d3.add_argument("--texture", type=int, choices=[1024, 2048, 4096], default=4096, help="texture map size")
    d3.add_argument("--keep-background", action="store_true", help="do not cut the object out first")
    d3.add_argument("-n", "--count", type=int, default=1)
    d3.add_argument("--seed", type=int)

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
                                                     "seed": a.seed, "width": a.width, "height": a.height})
        elif a.cmd == "voice":
            _submit_and_fetch(a, "qwen-tts", {"texts": [a.text], "voice_description": a.description,
                                              "language": a.language, "count": a.count, "seed": a.seed})
        elif a.cmd == "speak":
            ref_text = Path(a.ref_text_file).read_text().strip() if a.ref_text_file else a.ref_text
            _submit_and_fetch(a, "qwen-tts", {"texts": a.texts, "reference_audio": a.ref, "reference_text": ref_text,
                                              "language": a.language, "seed": a.seed})
        elif a.cmd == "speak-multilingual":
            _submit_and_fetch(a, "confucius-tts", {"texts": a.texts, "reference_audio": a.ref,
                                                   "language": a.language, "seed": a.seed})
        elif a.cmd == "video":
            _submit_and_fetch(a, "ltx-video", {"prompt": a.prompt, "images": a.image, "duration_s": a.duration,
                                               "width": a.width, "height": a.height, "fps": a.fps,
                                               "enhance_prompt": a.enhance, "count": a.count, "seed": a.seed})
        elif a.cmd == "audio-to-video":
            _submit_and_fetch(a, "ltx-video", {"prompt": a.prompt, "audio": a.audio,
                                               "images": [a.image] if a.image else [], "duration_s": a.duration,
                                               "width": a.width, "height": a.height, "count": a.count,
                                               "seed": a.seed})
        elif a.cmd == "retake":
            _submit_and_fetch(a, "ltx-video", {"video": a.video, "prompt": a.prompt, "start_s": a.start,
                                               "end_s": a.end, "regenerate": a.regenerate, "count": a.count,
                                               "seed": a.seed})
        elif a.cmd == "3d":
            _submit_and_fetch(a, "pixal3d", {"image": a.image, "remove_background": not a.keep_background,
                                             "resolution": a.resolution, "max_faces": a.faces,
                                             "texture_size": a.texture, "count": a.count, "seed": a.seed})
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
