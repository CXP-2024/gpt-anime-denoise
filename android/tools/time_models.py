"""Controlled CPU timing: 6B vs compact v3, interleaved, best-of-N.

The dev box also runs other GPU/CPU work, so a single pass is noisy. Both models
are timed in the same process, alternating, and the best time per stage is kept
(best time is the cleanest estimate of the intrinsic cost).

  .venv\Scripts\python.exe android\tools\time_models.py [reps]
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
MODELS = HERE / "models"
NAMES = ["realesrgan_x4plus_anime_6b", "realesr_animevideov3_x4"]
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def session(name):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(MODELS / f"{name}.onnx"), o, providers=["CPUExecutionProvider"])


def sr(sess, small):
    x = (np.asarray(small, np.float32) / 255).transpose(2, 0, 1)[None]
    y = np.clip(sess.run(None, {"input": x})[0], 0, 1)[0].transpose(1, 2, 0)
    return Image.fromarray((y * 255 + 0.5).astype(np.uint8))


def main():
    sess = {n: session(n) for n in NAMES}
    src = dn.open_image(GOLDEN / "in_01ee25d8.png")[0]
    W, H = src.size
    small = src.resize((round(W * dn.DOWN), round(H * dn.DOWN)), Image.BOX)

    # Warm up both (oneDNN primitives, thread pool, allocator arena).
    for _ in range(3):
        for n in NAMES:
            sr(sess[n], small)

    best = {n: {"sr_s": 1e9, "resize_back_s": 1e9} for n in NAMES}
    for _ in range(REPS):
        for n in NAMES:
            t = time.perf_counter(); big = sr(sess[n], small); dt = time.perf_counter() - t
            t = time.perf_counter(); big.resize((W, H), Image.LANCZOS); db = time.perf_counter() - t
            best[n]["sr_s"] = min(best[n]["sr_s"], dt)
            best[n]["resize_back_s"] = min(best[n]["resize_back_s"], db)

    # Downscale and L0 (identical for both models, so time once).
    t = time.perf_counter()
    for _ in range(5):
        src.resize((round(W * dn.DOWN), round(H * dn.DOWN)), Image.BOX)
    t_down = (time.perf_counter() - t) / 5

    mid = np.load(GOLDEN / "mid_01ee25d8.npy")
    t_l0 = 1e9
    for _ in range(2):
        t = time.perf_counter(); dn.l0_smooth(mid, dn.L0, device="cpu"); t_l0 = min(t_l0, time.perf_counter() - t)

    res = dict(reps=REPS, input="512x512", downscaled=list(small.size), models={})
    print(f"input {W}x{H} -> downscaled {small.size} -> model {small.size[0]*4}x{small.size[1]*4} -> back {W}x{H}")
    print(f"downscale (PIL BOX) : {t_down*1000:.2f} ms")
    print(f"L0 float64, 26 iter : {t_l0*1000:.0f} ms")
    for n in NAMES:
        v = best[n]
        total = t_down + v["sr_s"] + v["resize_back_s"] + t_l0
        res["models"][n] = dict(model_mb=round((MODELS / f"{n}.onnx").stat().st_size / 1e6, 3),
                                sr_s=round(v["sr_s"], 3), resize_back_s=round(v["resize_back_s"], 4),
                                downscale_s=round(t_down, 5), l0_s=round(t_l0, 3), total_s=round(total, 3))
        print(f"{n:32s} SR {v['sr_s']:.2f} s | back {v['resize_back_s']*1000:.1f} ms | total {total:.2f} s")
    res["speedup_v3_vs_6b"] = round(best[NAMES[0]]["sr_s"] / best[NAMES[1]]["sr_s"], 2)
    print(f"v3 SR speedup over 6B: {res['speedup_v3_vs_6b']}x")
    (HERE / "timing_models.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


main()
