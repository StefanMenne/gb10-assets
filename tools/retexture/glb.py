"""Replace the base color texture of a GLB in place (standard library only): geometry, UVs, other maps, skins and
animations stay byte for byte. Only the image data of the material's base color image changes; with
reset_pbr the metallic-roughness texture is dropped for plain factors (a new paint no longer matches it)."""
import json
import struct

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942


def read(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2:
        raise ValueError(f"{path} is not a binary glTF 2.0 file (.glb)")
    off, doc, binary = 12, None, b""
    while off < len(data):
        length, kind = struct.unpack_from("<II", data, off)
        chunk = data[off + 8:off + 8 + length]
        if kind == JSON_CHUNK:
            doc = json.loads(chunk)
        elif kind == BIN_CHUNK:
            binary = chunk
        off += 8 + length
    return doc, binary


def write(path, doc, binary):
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(js) + (8 + len(binary) if binary else 0)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(js), JSON_CHUNK) + js)
        if binary:
            f.write(struct.pack("<II", len(binary), BIN_CHUNK) + binary)


def base_color_image(doc):
    """Index of the one image all materials use as base color texture."""
    images = set()
    for m in doc.get("materials", []):
        t = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if t is not None:
            images.add(doc["textures"][t["index"]]["source"])
    if len(images) != 1:
        raise ValueError(f"the model needs exactly one base color texture, it has {len(images)}")
    return images.pop()


def image_bytes(doc, binary, image):
    v = doc["bufferViews"][doc["images"][image]["bufferView"]]
    return binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]


def replace_images(doc, binary, new):
    """new: {image index: PNG bytes}. Rebuilds the binary chunk with every buffer view at its old position
    order, the replaced ones with their new data."""
    views = doc["bufferViews"]
    repl = {}
    for image, data in new.items():
        img = doc["images"][image]
        if "bufferView" not in img:
            raise ValueError("images outside the GLB (uri) are not supported")
        repl[img["bufferView"]] = data
        img["mimeType"] = "image/png"
    out = bytearray()
    for k in sorted(range(len(views)), key=lambda k: views[k].get("byteOffset", 0)):
        v = views[k]
        if v.get("buffer", 0) != 0:
            continue
        data = repl.get(k, binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]])
        out += b"\0" * (-len(out) % 8)   # keeps every accessor aligned
        v["byteOffset"] = len(out)
        v["byteLength"] = len(data)
        out += data
    doc["buffers"][0]["byteLength"] = len(out)
    return bytes(out)


def retexture(src, dst, png, reset_pbr=True, roughness=0.8, tiny_png=None):
    """Write dst: src with the base color image replaced by png (bytes). reset_pbr: no metallic-roughness
    texture, metallic 0 and the given roughness; its image shrinks to tiny_png if nothing else uses it."""
    doc, binary = read(src)
    new = {base_color_image(doc): png}
    if reset_pbr:
        dropped = set()
        for m in doc.get("materials", []):
            pbr = m.setdefault("pbrMetallicRoughness", {})
            t = pbr.pop("metallicRoughnessTexture", None)
            if t is not None:
                dropped.add(doc["textures"][t["index"]]["source"])
            pbr["metallicFactor"] = 0.0
            pbr["roughnessFactor"] = roughness
        used = {doc["textures"][t["index"]]["source"] for m in doc.get("materials", []) for t in _texture_refs(m)}
        for image in dropped - used:
            if tiny_png is not None and image not in new:
                new[image] = tiny_png
    binary = replace_images(doc, binary, new)
    write(dst, doc, binary)


def _texture_refs(m):
    pbr = m.get("pbrMetallicRoughness", {})
    for t in (pbr.get("baseColorTexture"), pbr.get("metallicRoughnessTexture"), m.get("normalTexture"),
              m.get("occlusionTexture"), m.get("emissiveTexture")):
        if t is not None:
            yield t
