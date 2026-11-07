"""Remove the fish-scale texture that GPT image models (gpt-image-2 / 2.5 / 2.6, ChatGPT, Codex image_gen) leave in
anime and TV-anime style images.

Default method (recommended): downscale to 60% -> Real-ESRGAN x4plus anime 6B -> resize back -> L0 smoothing 0.001.
With an NVIDIA GPU both steps run on it (about 2 s for a 1.6 MP image); without one everything runs on the CPU.

  python denoise.py image.png [more.png ...]            # default method only
  python denoise.py image.png --all                      # all 7 candidates + a 4x2 comparison sheet
  python denoise.py ./folder --out ./cleaned             # every image in a folder
  python denoise.py image.png --l0 0.002                 # stronger L0 (0.001 is only the recommended value)

Outputs keep the input size. Models are downloaded on first use into --models (default ~/.cache/gpt-anime-denoise).
"""
import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

MODELS = {
    "esrgan": ("RealESRGAN_x4plus_anime_6B.pth",
               "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
    "apisr": ("2x_APISR_RRDB_GAN_generator.pth",
              "https://huggingface.co/camenduru/APISR/resolve/main/2x_APISR_RRDB_GAN_generator.pth"),
    "cugan": ("up2x-latest-denoise3x.pth",
              "https://github.com/bilibili/ailab/releases/download/Real-CUGAN/updated_weights.zip"),
}
SHA256 = {
    "RealESRGAN_x4plus_anime_6B.pth": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    "2x_APISR_RRDB_GAN_generator.pth": "3b0d2b3a3c0461ac17d00f4f32240666fb832b738ea5a48449b1acf07fbb07e5",
    "up2x-latest-denoise3x.pth": "0a14739f3f5fcbd74ec3ce2806d13a47916c916b20afe4a39d95f6df4ca6abd8",
}
DOWN = 0.6
L0 = 0.001  # recommended default; tune with --l0 (larger = flatter, too large wipes soft blush/gradients)
METHODS = ["esrgan60_l0", "esrgan60", "cugan60_l0", "cugan60", "apisr", "l0_0.001", "l0_0.0005"]
LABELS = {"original": "original", "l0_0.0005": "L0 0.0005", "l0_0.001": "L0 0.001", "apisr": "APISR",
          "cugan60": "CUGAN @60%", "cugan60_l0": "CUGAN @60% + L0 {lam:g}",
          "esrgan60": "ESRGAN @60%", "esrgan60_l0": "ESRGAN @60% + L0 {lam:g} (default)"}
SHEET_ORDER = ["original", "l0_0.0005", "l0_0.001", "apisr", "cugan60", "cugan60_l0", "esrgan60", "esrgan60_l0"]


def fetch(key, folder):
    name, url = MODELS[key]
    path = folder / name
    if path.exists():
        return path
    folder.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ...", flush=True)
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "gpt-anime-denoise"})).read()
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(next(n for n in z.namelist() if n.endswith(name)))
    if hashlib.sha256(data).hexdigest() != SHA256[name]:
        raise SystemExit(f"checksum mismatch for {name} from {url}; not saved")
    path.write_bytes(data)
    return path


def load_model(key, folder, device):
    import spandrel
    return spandrel.ModelLoader().load_from_file(str(fetch(key, folder))).model.to(device).eval()


class Cancelled(Exception):
    """Raised when the optional stop() callback asks a long step to give up early."""


def _check(stop):
    if stop is not None and stop():
        raise Cancelled()


def l0_smooth(img, lam, kappa=2.0, beta_max=1e5, device="cpu", stop=None):
    """L0 gradient minimization (Xu et al. 2011) on an HxWx3 float image in [0, 1].

    Flattens small ripples inside flat colour areas while keeping lines and colour edges; invents no new detail.
    On a CUDA device it runs the same float64 maths with torch.fft (~30x faster, identical 8-bit output).
    stop: optional callable checked every iteration; when it returns True, Cancelled is raised.
    """
    if str(device).startswith("cuda"):
        try:
            return _l0_torch(img, lam, kappa, beta_max, device, stop)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()  # very large image: fall back to the CPU path
    S = img.astype(np.float64)
    H, W, _ = S.shape
    fx = np.zeros((H, W)); fx[0, 0], fx[0, -1] = -1, 1
    fy = np.zeros((H, W)); fy[0, 0], fy[-1, 0] = -1, 1
    otf = np.abs(np.fft.fft2(fx)) ** 2 + np.abs(np.fft.fft2(fy)) ** 2
    N1 = np.fft.fft2(S, axes=(0, 1))
    beta = 2 * lam
    while beta < beta_max:
        _check(stop)
        h = np.roll(S, -1, 1) - S
        v = np.roll(S, -1, 0) - S
        t = (h ** 2 + v ** 2).sum(2) < lam / beta
        h[t] = 0
        v[t] = 0
        div = (np.roll(h, 1, 1) - h) + (np.roll(v, 1, 0) - v)
        S = np.real(np.fft.ifft2((N1 + beta * np.fft.fft2(div, axes=(0, 1))) / (1 + beta * otf)[..., None], axes=(0, 1)))
        beta *= kappa
    return S.clip(0, 1)


def _l0_torch(img, lam, kappa, beta_max, device, stop=None):
    S = torch.from_numpy(np.ascontiguousarray(img, np.float64)).to(device).permute(2, 0, 1)  # 3,H,W
    _, H, W = S.shape
    fx = torch.zeros(H, W, dtype=torch.float64, device=device); fx[0, 0], fx[0, -1] = -1, 1
    fy = torch.zeros(H, W, dtype=torch.float64, device=device); fy[0, 0], fy[-1, 0] = -1, 1
    otf = torch.fft.fft2(fx).abs() ** 2 + torch.fft.fft2(fy).abs() ** 2
    N1 = torch.fft.fft2(S)
    beta = 2 * lam
    while beta < beta_max:
        _check(stop)
        h = torch.roll(S, -1, 2) - S
        v = torch.roll(S, -1, 1) - S
        t = (h ** 2 + v ** 2).sum(0, keepdim=True) < lam / beta
        h = h.masked_fill(t, 0)
        v = v.masked_fill(t, 0)
        div = (torch.roll(h, 1, 2) - h) + (torch.roll(v, 1, 1) - v)
        S = torch.fft.ifft2((N1 + beta * torch.fft.fft2(div)) / (1 + beta * otf)).real
        beta *= kappa
    out = S.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    del S, N1, otf
    return out


def to_arr(im):
    return np.asarray(im.convert("RGB"), np.float64) / 255


def to_img(a):
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def _upscale_whole(model, im, device):
    t = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        y = model(t).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((y * 255 + 0.5).astype(np.uint8))


SR_BYTES_PER_PIXEL = 15000  # measured: the x4 anime 6B model peaks near 8.7 GB for a 0.57 MP input


def _tile_size(im, device):
    """Tile edge to use, or None when the whole image fits in free GPU memory.

    Planning ahead matters on Windows: the NVIDIA driver spills into system RAM instead of raising
    OutOfMemoryError, so an oversized whole-image pass just crawls for minutes.
    """
    if not str(device).startswith("cuda"):
        return None
    free, _ = torch.cuda.mem_get_info()
    if im.width * im.height * SR_BYTES_PER_PIXEL < free * 0.8:
        return None
    side = int((free * 0.25 / SR_BYTES_PER_PIXEL) ** 0.5) - 64  # leave headroom: near-full VRAM is slow
    return max(128, min(512, side))


def upscale(model, im, device, tile=256, border=32, stop=None):
    """Whole image at once when it fits; otherwise the same network over overlapping tiles."""
    planned = _tile_size(im, device)
    if planned:
        tile = planned
    else:
        try:
            return _upscale_whole(model, im, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    W, H = im.size
    out, scale = None, None
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            _check(stop)
            x2, y2 = min(x + tile, W), min(y + tile, H)
            box = (max(0, x - border), max(0, y - border), min(W, x2 + border), min(H, y2 + border))
            patch = _upscale_whole(model, im.crop(box), device)
            if out is None:
                scale = patch.width // (box[2] - box[0])
                out = Image.new("RGB", (W * scale, H * scale))
            out.paste(patch.crop(((x - box[0]) * scale, (y - box[1]) * scale,
                                  (x2 - box[0]) * scale, (y2 - box[1]) * scale)), (x * scale, y * scale))
    return out


def sr_at(model, src, device, stop=None):
    """Downscale (box filter averages the fine texture away), upscale with the model, resize back to the input size."""
    W, H = src.size
    small = src.resize((round(W * DOWN), round(H * DOWN)), Image.BOX)
    return upscale(model, small, device, stop=stop).resize((W, H), Image.LANCZOS)


def run(src, methods, models, device, lam=L0):
    W, H = src.size
    out, cache = {}, {}
    for m in methods:
        if m.startswith("l0_"):
            out[m] = to_img(l0_smooth(to_arr(src), float(m[3:]), device=device))
        elif m == "apisr":
            out[m] = upscale(models["apisr"], src, device).resize((W, H), Image.LANCZOS)
        else:
            key = m.split("60")[0]
            if key not in cache:
                cache[key] = sr_at(models[key], src, device)
            out[m] = to_img(l0_smooth(to_arr(cache[key]), lam, device=device)) if m.endswith("_l0") else cache[key]
    return out


def sheet(src, res, path, lam=L0):
    items = [("original", src)] + [(k, res[k]) for k in SHEET_ORDER[1:]]
    w, h = src.size
    g, top, cols = 20, max(40, w // 18), 4
    s = Image.new("RGB", (cols * (w + g) - g, 2 * (h + top + g)), (60, 60, 60))
    d = ImageDraw.Draw(s)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", top // 2)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", top // 2)
        except OSError:
            font = ImageFont.load_default()
    for i, (k, im) in enumerate(items):
        x, y = (i % cols) * (w + g), (i // cols) * (h + top + g)
        s.paste(im, (x, y + top))
        d.text((x + 12, y + top // 5), LABELS[k].format(lam=lam), fill="white", font=font)
    s.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="images or folders")
    ap.add_argument("--out", help="output folder (default: <image folder>/<stem>_denoise)")
    ap.add_argument("--all", action="store_true", help="write all 7 candidates and a comparison sheet")
    ap.add_argument("--method", default="esrgan60_l0", choices=METHODS)
    ap.add_argument("--l0", type=float, default=L0,
                    help="L0 strength for the *_l0 methods (default 0.001; try 0.0005-0.002)")
    ap.add_argument("--models", default=str(Path.home() / ".cache" / "gpt-anime-denoise"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    files = []
    for p in map(Path, a.inputs):
        files += sorted(q for q in p.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")) if p.is_dir() else [p]
    methods = METHODS if a.all else [a.method]
    need = {m.split("60")[0] if "60" in m else m for m in methods if not m.startswith("l0_")}
    models = {k: load_model(k, Path(a.models), a.device) for k in need}

    for p in files:
        src = Image.open(p).convert("RGB")
        out = Path(a.out) if a.out else p.parent / f"{p.stem}_denoise"
        out.mkdir(parents=True, exist_ok=True)
        res = run(src, methods, models, a.device, a.l0)
        for k, im in res.items():
            im.save(out / f"{p.stem}_{k}.png")
        if a.all:
            sheet(src, res, out / f"{p.stem}_sheet.png", a.l0)
        print(f"{p} -> {out}", flush=True)


if __name__ == "__main__":
    main()
