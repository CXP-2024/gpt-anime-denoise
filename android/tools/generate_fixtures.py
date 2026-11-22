
"""Build the golden test fixtures for the Android port with denoise.py's exact code path."""
import json, sys, time
from pathlib import Path

ROOT = Path(r"F:\gpt-anime-denoise")
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import denoise as dn

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
SRC_DIR = ROOT / "outputs"
MODEL_CACHE = Path(r"C:\Users\21294\.cache\gpt-anime-denoise")
CROP = 512
PICKS = [
    ("01ee25d8", "img_01ee25d8ded0d126f4bc8b0a00b2f6c0-1_denoise.png"),
    ("061d63a2", "img_061d63a2c2a2c6fceea865c2fa78e1d6-1_denoise.png"),
    ("4dde2cb0", "img_4dde2cb047cc18a1b9c27fed8e344d53-1_denoise.png"),
]

def crop_of(im):
    """Deterministic 512x512 crop: horizontally centred, vertical centre at 38% of the height."""
    W, H = im.size
    s = min(CROP, W, H)
    x0 = (W - s) // 2
    y0 = max(0, min(H - s, int(H * 0.38) - s // 2))
    return im.crop((x0, y0, x0 + s, y0 + s)), (x0, y0)

def main():
    GOLDEN.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model = dn.load_model("esrgan", MODEL_CACHE, "cpu")
    print(f"model loaded in {time.perf_counter() - t0:.2f}s", flush=True)
    dn.sr_at(model, Image.new("RGB", (64, 64), (128, 128, 128)), "cpu")  # warm-up
    rows = []
    for tag, name in PICKS:
        im = dn.open_image(SRC_DIR / name)[0]
        crop, origin = crop_of(im)
        inp = GOLDEN / f"in_{tag}.png"
        crop.save(inp, "PNG")
        src = dn.open_image(inp)[0]                      # re-open the saved bytes
        W, H = src.size
        print(f"[{tag}] {name} {im.size} -> crop {crop.size} at {origin}", flush=True)

        t = time.perf_counter(); small = src.resize((round(W * dn.DOWN), round(H * dn.DOWN)), Image.BOX); t_down = time.perf_counter() - t
        t = time.perf_counter(); big = dn.upscale(model, small, "cpu"); t_esr = time.perf_counter() - t
        t = time.perf_counter(); sr = big.resize((W, H), Image.LANCZOS); t_back = time.perf_counter() - t
        if tag == PICKS[0][0]:
            ref = dn.to_arr(dn.sr_at(model, src, "cpu"))
            assert np.array_equal(ref, dn.to_arr(sr)), "inline pipeline diverged from dn.sr_at()"
            print("  sr_at() cross-check: bit-identical", flush=True)

        mid = dn.to_arr(sr)
        np.save(GOLDEN / f"mid_{tag}.npy", mid)
        t = time.perf_counter(); sm = dn.l0_smooth(mid, dn.L0, device="cpu"); t_l0 = time.perf_counter() - t
        out = GOLDEN / f"out_{tag}.png"
        dn.save_png(dn.to_img(sm), out)
        assert np.array_equal(np.asarray(Image.open(out).convert("RGB")), np.asarray(dn.to_img(sm)))
        assert mid.dtype == np.float64 and mid.shape == (H, W, 3)
        rows.append(dict(tag=tag, source=name, size=[W, H], crop_origin=list(origin),
                         t_downscale_s=round(t_down, 4), t_esrgan_s=round(t_esr, 3),
                         t_resize_back_s=round(t_back, 4), t_l0_s=round(t_l0, 4),
                         mid_min=float(mid.min()), mid_max=float(mid.max()), mid_mean=float(mid.mean())))
        print(f"  down {t_down*1000:.1f} ms | esrgan {t_esr:.2f} s | back {t_back*1000:.1f} ms | l0 {t_l0*1000:.1f} ms", flush=True)
    (GOLDEN / "timing.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

main()
