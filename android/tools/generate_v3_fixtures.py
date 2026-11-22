"""Build reference fixtures for the OPTIONAL compact model (realesr-animevideov3).

`denoise.py` has no method that uses this network, so this is not a port of an
existing code path - it is the same pipeline with the smaller network swapped in:

    small = src.resize((round(W*DN_DOWN), round(H*DN_DOWN)), Image.BOX)
    big   = onnxruntime(realesr_animevideov3_x4.onnx, small)      # 4x
    sr    = big.resize((W, H), Image.LANCZOS)
    mid   = denoise.to_arr(sr)
    out   = denoise.to_img(denoise.l0_smooth(mid, 0.001, device="cpu"))

Writes v3_mid_<tag>.npy and v3_out_<tag>.png next to the 6B fixtures. The
downscale, resize-back and L0 stages are literally denoise.py's own functions;
only the model differs. Everything else (crop, tags) matches the 6B fixtures so
the same input PNG is reused.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"F:\gpt-anime-denoise")
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import onnxruntime as ort

import denoise as dn

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
ONNX = HERE / "models" / "realesr_animevideov3_x4.onnx"
TAGS = ["01ee25d8", "061d63a2", "4dde2cb0"]


def session():
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(ONNX), o, providers=["CPUExecutionProvider"])


def upscale(sess, small):
    x = (np.asarray(small, np.float32) / 255).transpose(2, 0, 1)[None]
    y = np.clip(sess.run(None, {"input": x})[0], 0, 1)[0].transpose(1, 2, 0)
    return Image.fromarray((y * 255 + 0.5).astype(np.uint8))


def main():
    sess = session()
    upscale(sess, Image.new("RGB", (64, 64), (128, 128, 128)))      # warm-up
    times = []
    for tag in TAGS:
        src = dn.open_image(GOLDEN / f"in_{tag}.png")[0]
        W, H = src.size
        t = time.perf_counter()
        small = src.resize((round(W * dn.DOWN), round(H * dn.DOWN)), Image.BOX)
        sr = upscale(sess, small).resize((W, H), Image.LANCZOS)
        t_sr = time.perf_counter() - t
        mid = dn.to_arr(sr)
        np.save(GOLDEN / f"v3_mid_{tag}.npy", mid)
        t = time.perf_counter()
        sm = dn.l0_smooth(mid, dn.L0, device="cpu")
        t_l0 = time.perf_counter() - t
        out = GOLDEN / f"v3_out_{tag}.png"
        dn.save_png(dn.to_img(sm), out)
        assert np.array_equal(np.asarray(Image.open(out).convert("RGB")), np.asarray(dn.to_img(sm)))
        times.append(dict(tag=tag, t_sr_stage_s=round(t_sr, 3), t_l0_s=round(t_l0, 3)))
        print(f"[{tag}] sr {t_sr:.2f} s | l0 {t_l0:.2f} s -> v3_mid_{tag}.npy + v3_out_{tag}.png", flush=True)
    (HERE / "v3_timing.json").write_text(json.dumps(times, indent=2) + "\n", encoding="utf-8")


main()