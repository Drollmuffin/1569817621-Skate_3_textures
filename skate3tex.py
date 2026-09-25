#!/usr/bin/env python3
"""
skate3tex - automatic Skate 3 (PS3) texture maker.

Turns any picture (PNG, JPG, WEBP, GIF, BMP...) into a ready-to-use Skate 3
.psg texture, and turns .psg textures back into PNGs so you can edit them.

Format handled: RenderWare 4 PS3 texture ("RW4ps3"), 512x512, DXT5, 10 mipmaps
- the same format as the deck/graphic textures in this repo.

USAGE
  python skate3tex.py                     batch: every image in ./input -> ./output/*.psg
  python skate3tex.py pic.png             one image -> pic.psg (next to it)
  python skate3tex.py tex.psg             one texture -> tex.png (to preview/edit)
  python skate3tex.py a.png b.jpg c.psg   any mix of files (drag & drop works too)

OPTIONS
  -o, --out DIR        where to write results (default: next to input / ./output)
  --fit MODE           how non-square pictures become 512x512:
                         stretch  squash to fit, nothing cut off
                         cover    fill the square, crop the overflow (default)
                         contain  fit inside, pad with transparent borders
  --name NAME          output file name (single input only), e.g. 106508011
                       so it directly replaces that texture in the game files
  --template FILE.psg  copy the header from this .psg instead of the built-in one
  --alpha N|keep       alpha channel to store (default 25, the value every
                       original Skate 3 texture here uses; the game treats it
                       as a mask, not transparency). "keep" keeps the picture's
                       own alpha. Transparent areas are filled with --bg.
  --bg COLOR           fill colour behind transparent areas (default black)
  --preview            also save a PNG of how the generated texture decodes
  --watch              keep running and auto-convert anything dropped into ./input
"""

import argparse
import base64
import os
import sys
import time

try:
    import numpy as np
    from PIL import Image
except ImportError:
    sys.exit("Missing libraries. Run:  pip install pillow numpy")

SIZE = 512
MIP_COUNT = 10
HEADER_SIZE = 0x238
DATA_SIZE = sum(max(1, (SIZE >> i) // 4) ** 2 * 16 for i in range(MIP_COUNT))  # 349552
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tga", ".tif", ".tiff"}

# Header shared by every texture in this repo (RW4ps3, DXT5 0x88, 512x512, 10 mips).
TEMPLATE_HEADER = base64.b64decode(
    "iVJXNHBzMwANChoKASAEADQ1NAAwMDAAAAAAAIPI/ekAAAAEAAAABAAAABAAAAAAAAAB2AAAAMAAAAAAAAAAAAAAAAAAAAI4AAAA"
    "EAAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAABAAVVcAAAAIAAAAOQAAAABAAAAAAAAAABAAAAAAAAAAEAAAAAAAAAAQAA"
    "AAAAAAABAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEABAAAAAQAAAAMAAAAHAAAAFAAAAB0AAAAkAABAAUA"
    "AAAKAAAADAAAAAAAAQAwAAEAMQABADIAAQAzAAEANAABABAAAgDoAOsACADrAAsAAQAGAAAAAwAAABiDyP3p/7AAAIPI/ekAAAAA"
    "AAAAAAAAAAAAAQAHAAAAAAAAAAAAAAAAAAACOAAAAjgAAAAAAAEACAAAAAAAAAAAiAoCAAAAquQCAAIAAAEAAAAACAAAAAAAAAAA"
    "AAAAAAIAAAAAVXIAiAAAAAAAAAAAAAAAAAAAAAEAAAAUAAAALAAAAAAAAAA4AAAALJsPFnh9GZyksMPIbqxGLkoAAAABMjA2LlRl"
    "eHR1cmUAAAAAAAAAAAAAAAAZAAAABAAAAAAAAAAAAAVVcAAAAIAAAAAFAAEANAAAAVwAAAAAAAAAKAAAAAQAAAAHAAIA6AAAAZAA"
    "AAAAAAAAOAAAABAAAAAJAOsACwAAAdAAAAAAAAAACAAAABAAAAAIAOsACA=="
)
assert len(TEMPLATE_HEADER) == HEADER_SIZE


# ---------------------------------------------------------------- DXT5 encode

def _to_blocks(img):
    """(H, W, 4) uint8 -> (N, 16, 4) float32 of 4x4 blocks in row-major order."""
    h, w, _ = img.shape
    b = img.reshape(h // 4, 4, w // 4, 4, 4).transpose(0, 2, 1, 3, 4)
    return b.reshape(-1, 16, 4).astype(np.float32)


def _pack565(c):
    c = np.clip(np.rint(c), 0, 255).astype(np.uint32)
    return ((c[:, 0] * 31 + 127) // 255 << 11) | ((c[:, 1] * 63 + 127) // 255 << 5) | ((c[:, 2] * 31 + 127) // 255)


def _unpack565(v):
    r = (v >> 11) & 31
    g = (v >> 5) & 63
    b = v & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.float32)


def _encode_color(rgb):
    """rgb: (N, 16, 3) float -> (N, 8) uint8 DXT color blocks."""
    n = rgb.shape[0]
    mean = rgb.mean(1, keepdims=True)
    centered = rgb - mean
    cov = np.einsum("nki,nkj->nij", centered, centered)
    # Principal axis via power iteration, starting from the bounding-box diagonal.
    axis = rgb.max(1) - rgb.min(1) + 1e-3
    for _ in range(8):
        axis = np.einsum("nij,nj->ni", cov, axis)
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
    proj = np.einsum("nki,ni->nk", centered, axis)
    idx = np.arange(n)
    hi = rgb[idx, proj.argmax(1)]
    lo = rgb[idx, proj.argmin(1)]
    # Inset endpoints slightly to reduce quantisation error.
    inset = (hi - lo) / 16.0
    c0 = _pack565(hi - inset)
    c1 = _pack565(lo + inset)
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)

    e0, e1 = _unpack565(c0), _unpack565(c1)
    palette = np.stack([e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3], 1)  # (N, 4, 3)
    dist = ((rgb[:, :, None, :] - palette[:, None, :, :]) ** 2).sum(-1)
    sel = dist.argmin(-1).astype(np.uint32)
    sel[c0 == c1] = 0  # flat block: 3-colour mode would be used otherwise

    bits = (sel << (2 * np.arange(16, dtype=np.uint32))).sum(1, dtype=np.uint32)
    out = np.zeros((n, 8), np.uint8)
    out[:, 0], out[:, 1] = c0 & 0xFF, c0 >> 8
    out[:, 2], out[:, 3] = c1 & 0xFF, c1 >> 8
    for i in range(4):
        out[:, 4 + i] = (bits >> (8 * i)) & 0xFF
    return out


def _encode_alpha(a):
    """a: (N, 16) float -> (N, 8) uint8 DXT5 alpha blocks."""
    n = a.shape[0]
    a0 = a.max(1).astype(np.int64)
    a1 = a.min(1).astype(np.int64)
    # 8-alpha mode (a0 > a1): index 0=a0, 1=a1, 2..7 interpolated.
    t = np.array([0, 7, 1, 2, 3, 4, 5, 6], np.float32) / 7.0
    palette = a0[:, None] * (1 - t) + a1[:, None] * t
    sel = np.abs(a[:, :, None] - palette[:, None, :]).argmin(-1).astype(np.uint64)
    sel[a0 == a1] = 0
    bits = (sel << (3 * np.arange(16, dtype=np.uint64))).sum(1, dtype=np.uint64)
    out = np.zeros((n, 8), np.uint8)
    out[:, 0], out[:, 1] = a0, a1
    for i in range(6):
        out[:, 2 + i] = ((bits >> np.uint64(8 * i)) & np.uint64(0xFF)).astype(np.uint8)
    return out


def encode_dxt5(img):
    """(H, W, 4) uint8 RGBA, H and W multiples of 4 -> DXT5 bytes."""
    blocks = _to_blocks(img)
    return np.concatenate([_encode_alpha(blocks[:, :, 3]), _encode_color(blocks[:, :, :3])], 1).tobytes()


# ---------------------------------------------------------------- DXT5 decode

def decode_dxt5(data, w, h):
    bw, bh = max(1, w // 4), max(1, h // 4)
    raw = np.frombuffer(data, np.uint8, bw * bh * 16).reshape(-1, 16)
    n = raw.shape[0]

    a0 = raw[:, 0].astype(np.float32)
    a1 = raw[:, 1].astype(np.float32)
    abits = np.zeros(n, np.uint64)
    for i in range(6):
        abits |= raw[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    asel = ((abits[:, None] >> (np.uint64(3) * np.arange(16, dtype=np.uint64))) & np.uint64(7)).astype(np.int64)
    k = np.arange(8, dtype=np.float32)
    mode8 = np.where(k == 0, a0[:, None], np.where(k == 1, a1[:, None],
                     ((8 - k) * a0[:, None] + (k - 1) * a1[:, None]) / 7))
    mode6 = np.where(k == 0, a0[:, None], np.where(k == 1, a1[:, None],
                     np.where(k == 6, 0, np.where(k == 7, 255, ((6 - k) * a0[:, None] + (k - 1) * a1[:, None]) / 5))))
    apal = np.where((a0 > a1)[:, None], mode8, mode6)
    alpha = np.take_along_axis(apal, asel, 1)

    c0 = raw[:, 8].astype(np.uint32) | raw[:, 9].astype(np.uint32) << 8
    c1 = raw[:, 10].astype(np.uint32) | raw[:, 11].astype(np.uint32) << 8
    e0, e1 = _unpack565(c0), _unpack565(c1)
    cpal = np.stack([e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3], 1)
    cbits = raw[:, 12:16].astype(np.uint32)
    cbits = cbits[:, 0] | cbits[:, 1] << 8 | cbits[:, 2] << 16 | cbits[:, 3] << 24
    csel = (cbits[:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3
    rgb = np.take_along_axis(cpal, csel[:, :, None].astype(np.int64).repeat(3, 2), 1)

    px = np.concatenate([rgb, alpha[:, :, None]], 2)
    px = np.clip(np.rint(px), 0, 255).astype(np.uint8)
    img = px.reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return img[:h, :w]


# ---------------------------------------------------------------- conversions

def fit_square(im, mode):
    im = im.convert("RGBA")
    if mode == "stretch" or im.size == (SIZE, SIZE):
        return im.resize((SIZE, SIZE), Image.LANCZOS)
    w, h = im.size
    if mode == "cover":
        s = SIZE / min(w, h)
        im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.LANCZOS)
        l, t = (im.width - SIZE) // 2, (im.height - SIZE) // 2
        return im.crop((l, t, l + SIZE, t + SIZE))
    s = SIZE / max(w, h)  # contain
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
    return canvas


def image_to_psg_bytes(im, fit="cover", header=TEMPLATE_HEADER, alpha=25, bg="black"):
    base = fit_square(im, fit)
    if alpha != "keep":
        flat = Image.new("RGBA", base.size, bg)
        flat.alpha_composite(base)
        flat.putalpha(int(alpha))
        base = flat
    chunks = []
    for level in range(MIP_COUNT):
        dim = SIZE >> level
        mip = base if level == 0 else base.resize((dim, dim), Image.LANCZOS)
        arr = np.asarray(mip, np.uint8)
        if dim < 4:  # tiny mips still occupy one full 4x4 block
            arr = np.pad(arr, ((0, 4 - dim), (0, 4 - dim), (0, 0)), mode="edge")
        chunks.append(encode_dxt5(arr))
    data = b"".join(chunks)
    assert len(data) == DATA_SIZE
    return header + data


def psg_to_image(path):
    with open(path, "rb") as f:
        blob = f.read()
    if not blob.startswith(b"\x89RW4ps3"):
        raise ValueError(f"{path} is not an RW4 PS3 file")
    if len(blob) != HEADER_SIZE + DATA_SIZE or blob[0x15C] != 0x88:
        raise ValueError(f"{path} is not a 512x512 DXT5 texture (unsupported layout)")
    return Image.fromarray(decode_dxt5(blob[HEADER_SIZE:], SIZE, SIZE), "RGBA")


def load_header(template):
    if not template:
        return TEMPLATE_HEADER
    with open(template, "rb") as f:
        head = f.read(HEADER_SIZE)
    if not head.startswith(b"\x89RW4ps3"):
        sys.exit(f"Template {template} is not an RW4 PS3 file")
    return head


def convert(path, out_dir, args, header):
    stem, ext = os.path.splitext(os.path.basename(path))
    ext = ext.lower()
    out_dir = out_dir or os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    name = args.name or stem
    if ext == ".psg":
        dst = os.path.join(out_dir, name + ".png")
        im = psg_to_image(path)
        if im.getchannel("A").getextrema()[0] == im.getchannel("A").getextrema()[1]:
            im = im.convert("RGB")  # uniform alpha is a game mask, drop it so the PNG looks right
        im.save(dst)
    elif ext in IMAGE_EXTS:
        dst = os.path.join(out_dir, name + ".psg")
        with Image.open(path) as im:
            blob = image_to_psg_bytes(im, args.fit, header, args.alpha, args.bg)
        with open(dst, "wb") as f:
            f.write(blob)
        if args.preview:
            psg_to_image(dst).convert("RGB").save(os.path.join(out_dir, name + "_preview.png"))
    else:
        print(f"  skip  {path} (unsupported file type)")
        return
    print(f"  ok    {path} -> {dst}")


def alpha_arg(v):
    if v == "keep":
        return v
    if v.isdigit() and 0 <= int(v) <= 255:
        return int(v)
    raise argparse.ArgumentTypeError("use a number 0-255 or 'keep'")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Automatic Skate 3 texture maker",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("files", nargs="*", help="images to turn into .psg, or .psg files to turn into .png")
    ap.add_argument("-o", "--out", help="output folder")
    ap.add_argument("--fit", choices=["stretch", "cover", "contain"], default="cover")
    ap.add_argument("--name", help="output file name (single input only)")
    ap.add_argument("--template", help=".psg to copy the header from")
    ap.add_argument("--alpha", default="25", type=alpha_arg)
    ap.add_argument("--bg", default="black")
    ap.add_argument("--preview", action="store_true", help="also write a decoded preview PNG")
    ap.add_argument("--watch", action="store_true", help="auto-convert new files in ./input")
    args = ap.parse_args()
    header = load_header(args.template)

    if args.name and len(args.files) > 1:
        sys.exit("--name only works with a single input file")

    if args.files:
        for p in args.files:
            try:
                convert(p, args.out, args, header)
            except Exception as e:  # keep going on bad files
                print(f"  FAIL  {p}: {e}")
        return

    in_dir = os.path.join(here, "input")
    out_dir = args.out or os.path.join(here, "output")
    os.makedirs(in_dir, exist_ok=True)
    done = {}
    print(f"Converting everything in {in_dir} -> {out_dir}")
    while True:
        for fn in sorted(os.listdir(in_dir)):
            p = os.path.join(in_dir, fn)
            if not os.path.isfile(p) or os.path.splitext(fn)[1].lower() not in IMAGE_EXTS | {".psg"}:
                continue
            mtime = os.path.getmtime(p)
            if done.get(p) == mtime:
                continue
            try:
                convert(p, out_dir, args, header)
            except Exception as e:
                print(f"  FAIL  {p}: {e}")
            done[p] = mtime
        if not args.watch:
            break
        time.sleep(1)
    if not done:
        print("  (input folder is empty - drop some pictures in it and run again)")


if __name__ == "__main__":
    main()
