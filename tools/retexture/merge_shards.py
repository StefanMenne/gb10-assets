"""Join sharded safetensors files into one (standard library only), for download.sh:

  python merge_shards.py OUT.safetensors SHARD.safetensors ...

ComfyUI loads a model from one file; Comfy-Org's Qwen-Image files hold the same tensors (names, shapes, dtypes,
bytes) as the diffusers shards of tools/qwen-image, so joining those saves downloading 58 GB again. The tensor
data is copied unchanged; the file is written under a temporary name and renamed when complete.
"""
import json
import os
import struct
import sys


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def main(out, shards):
    entries, sources = {}, []
    offset = 0
    for path in shards:
        n, h = header(path)
        h.pop("__metadata__", None)
        for name, t in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            a, b = t["data_offsets"]
            entries[name] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [offset, offset + b - a]}
            sources.append((path, 8 + n + a, b - a))
            offset += b - a
    js = json.dumps(entries, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 8)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(js)) + js)
        for path, start, length in sources:
            with open(path, "rb") as src:
                src.seek(start)
                while length:
                    chunk = src.read(min(length, 64 << 20))
                    f.write(chunk)
                    length -= len(chunk)
    os.replace(tmp, out)
    print(f"[merge] {len(entries)} tensors, {offset / 1e9:.1f} GB -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
