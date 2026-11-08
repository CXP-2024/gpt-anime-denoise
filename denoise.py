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
import os
import urllib.request
import zipfile
from pathlib import Path

# AMD ROCm: by default MIOpen benchmarks every convolution algorithm the first time it meets a new input size,
# single-threaded on the CPU while the GPU sits idle. On an RX 7700 XT that is ~90 s per new image size for the
# ESRGAN layers. FAST picks a kernel by heuristics instead (~0.3 s); each pass is a few ms slower, which is noise
# next to that. Must be set before torch loads MIOpen; has no effect on NVIDIA or CPU. An explicit value wins.
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps  # noqa: E402

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


def default_device():
    """CUDA, else Apple Metal (MPS), else the CPU; --device still overrides."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _is_oom(err):
    """CUDA raises its own OutOfMemoryError; MPS raises a RuntimeError that says the same thing."""
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def _clear(device):
    """Release the backend's cached blocks after a big step (a no-op on the CPU)."""
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    elif str(device).startswith("mps"):
        torch.mps.empty_cache()


class Cancelled(Exception):
    """Raised when the optional stop() callback asks a long step to give up early."""


def _check(stop):
    if stop is not None and stop():
        raise Cancelled()


def l0_smooth(img, lam, kappa=2.0, beta_max=1e5, device="cpu", stop=None):
    """L0 gradient minimization (Xu et al. 2011) on an HxWx3 float image in [0, 1].

    Flattens small ripples inside flat colour areas while keeping lines and colour edges; invents no new detail.
    On a CUDA device it runs the same float64 maths with torch.fft (~30x faster, identical 8-bit output); on MPS,
    which has no float64, it runs in float32 and a few pixels may differ from the CPU result by 1-2 levels.
    stop: optional callable checked every iteration; when it returns True, Cancelled is raised.
    """
    if str(device).startswith(("cuda", "mps")):
        try:
            return _l0_torch(img, lam, kappa, beta_max, device, stop)
        except Exception as err:
            if not _is_oom(err):
                raise
            _clear(device)  # very large image: fall back to the CPU path
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
    dt = torch.float32 if str(device).startswith("mps") else torch.float64  # Metal has no float64
    S = torch.from_numpy(np.ascontiguousarray(img, np.float64)).to(device=device, dtype=dt).permute(2, 0, 1)  # 3,H,W
    _, H, W = S.shape
    fx = torch.zeros(H, W, dtype=dt, device=device); fx[0, 0], fx[0, -1] = -1, 1
    fy = torch.zeros(H, W, dtype=dt, device=device); fy[0, 0], fy[-1, 0] = -1, 1
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


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif", ".tga")


def open_image(src):
    """Open an image the way a viewer shows it; returns (rgb, alpha or None, icc profile bytes or None).

    EXIF rotation is applied, 16-bit / 32-bit greyscale is scaled to 8-bit (a plain convert would clip it to white),
    CMYK is converted through its embedded profile when it has one, and transparency is kept separately.
    The ICC profile is returned only for RGB sources, where it still describes the pixel values.
    """
    im = Image.open(src)
    im.load()
    im = ImageOps.exif_transpose(im)
    icc = im.info.get("icc_profile")
    mode = im.mode
    if mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        a = np.asarray(im, np.float64)
        scale = 257.0 if a.max() > 255 else 1.0
        im = Image.fromarray(np.clip(a / scale + 0.5, 0, 255).astype(np.uint8), "L")
    alpha = None
    if im.mode in ("RGBA", "LA", "PA", "La", "RGBa") or "transparency" in im.info:
        a = im.convert("RGBA").getchannel("A")
        if a.getextrema() != (255, 255):  # only keep alpha that is actually used
            alpha = a
    if im.mode == "CMYK" and icc:
        try:
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                                           ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception:
            pass
    if mode not in ("RGB", "RGBA", "RGBX", "P"):
        icc = None  # a grey or CMYK profile does not describe the RGB we produce
    return im.convert("RGB"), alpha, icc


def save_png(im, path, alpha=None, icc=None, **kw):
    """Save as PNG with the source's alpha and colour profile put back."""
    if alpha is not None:
        im = im.copy(); im.putalpha(alpha)
    if icc:
        kw["icc_profile"] = icc
    im.save(path, "PNG", **kw)


# Full FP32 convolutions: with cuDNN's default TF32, the rounding depends on the input shape and 92 stacked layers
# amplify it (up to 3 levels vs the CPU result, 17 between band shapes). Without TF32 a banded pass equals a
# whole-image pass exactly and the GPU matches the CPU to within 1 level, for ~0.1 s more per image.
torch.backends.cudnn.allow_tf32 = False

# Exact memory-bounded super-resolution. Not tiling with guessed borders: every band carries enough real
# neighbour rows for the layers it runs, so each output pixel sees exactly the inputs a whole-image pass would.
# Measured on a 614x922 input: the trunk peaks at 1.8 GB, the two 3x3 convs at 4x size peak at 8.7 GB (2.2 GB
# input + 2.2 GB output + cuDNN workspace). So the trunk runs whole whenever it fits, the 4x tail runs in bands.
HEAD_HALO = 96      # input rows: the trunk stacks ~92 3x3 convs (6 RRDB x 3 RDB x 5 + first/last conv)
TAIL_HALO = 3       # input rows: the tail is 4 3x3 convs around two nearest 2x upsamples (needs 2)
HEAD_BYTES = 3500   # per input pixel for the trunk (1.8 GB / 0.57 MP, plus headroom)
TAIL_BYTES = 16000  # per input pixel of a tail band (64 channels at 4x: input + output + conv workspace)


def _free_bytes(device):
    if str(device).startswith("cuda"):
        return torch.cuda.mem_get_info()[0]
    if str(device).startswith("mps"):
        try:
            return torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()
        except Exception:  # older PyTorch lacks recommended_max_memory: Apple GPUs share system RAM, so use that
            pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return 4 << 30


def _rrdb_split(model):
    """(trunk, tail) of a plain RRDB ESRGAN, split at its first upsample; None for other architectures."""
    seq = getattr(model, "model", None)
    if getattr(model, "shuffle_factor", None) or not isinstance(seq, torch.nn.Sequential):
        return None
    ups = [i for i, m in enumerate(seq) if isinstance(m, torch.nn.Upsample)]
    return (seq[:ups[0]], seq[ups[0]:]) if ups else None


def _bands(n, step, halo):
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        yield r0, r1, max(0, r0 - halo), min(n, r1 + halo)


def _run_bands(fn, n, budget, per_row, halo, stop, device="cpu"):
    """Call fn(r0, r1, a, b) over exact bands sized to the memory budget; on out-of-memory, halve and redo."""
    rows = max(8, int(budget / max(per_row, 1)) - 2 * halo)
    while True:
        try:
            for r0, r1, a, b in _bands(n, rows, halo):
                _check(stop)
                fn(r0, r1, a, b)
            return
        except Exception as err:
            if not _is_oom(err):
                raise
            _clear(device)
            if rows <= 8:
                raise
            rows = max(8, rows // 2)


def upscale(model, im, device, stop=None):
    """Super-resolve with the result of a single whole-image pass but bounded GPU memory / RAM (see above)."""
    split = _rrdb_split(model)
    if split is None:  # other models (APISR, CUGAN in --all): whole image
        return _upscale_whole(model, im, device)
    trunk, tail = split
    W, H = im.size
    x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None]
    with torch.no_grad():
        budget = _free_bytes(device) * 0.6
        if H * W * HEAD_BYTES <= budget:
            f = trunk(x.to(device))
        else:  # very large image: trunk in exact bands, features gathered on the CPU
            f = torch.empty((1, 64, H, W), dtype=torch.float32)
            def head_band(r0, r1, a, b):
                f[:, :, r0:r1] = trunk(x[:, :, a:b].to(device))[:, :, r0 - a:r1 - a].cpu()
            _run_bands(head_band, H, budget, W * HEAD_BYTES, HEAD_HALO, stop, device)
        out = {}
        def tail_band(r0, r1, a, b):
            y = tail(f[:, :, a:b].to(device)).clamp_(0, 1)
            s = y.shape[2] // (b - a)
            if "img" not in out:
                out["img"] = np.empty((H * s, W * s, 3), np.uint8)
            y = y[0, :, (r0 - a) * s:(r1 - a) * s].permute(1, 2, 0).cpu().numpy()
            out["img"][r0 * s:r1 * s] = (y * 255 + 0.5).astype(np.uint8)
        _run_bands(tail_band, H, _free_bytes(device) * 0.6, W * TAIL_BYTES, TAIL_HALO, stop, device)
    del f
    _clear(device)
    return Image.fromarray(out["img"])


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
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()

    files = []
    for p in map(Path, a.inputs):
        files += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    methods = METHODS if a.all else [a.method]
    need = {m.split("60")[0] if "60" in m else m for m in methods if not m.startswith("l0_")}
    models = {k: load_model(k, Path(a.models), a.device) for k in need}

    for p in files:
        src, alpha, icc = open_image(p)
        out = Path(a.out) if a.out else p.parent / f"{p.stem}_denoise"
        out.mkdir(parents=True, exist_ok=True)
        res = run(src, methods, models, a.device, a.l0)
        for k, im in res.items():
            save_png(im, out / f"{p.stem}_{k}.png", alpha, icc)
        if a.all:
            sheet(src, res, out / f"{p.stem}_sheet.png", a.l0)
        print(f"{p} -> {out}", flush=True)


if __name__ == "__main__":
    main()
