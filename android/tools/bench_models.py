
"""Benchmark + quality-compare the two SR models with onnxruntime, on the golden inputs.

Runs the *identical* pipeline as denoise.sr_at() (PIL BOX downscale at DOWN=0.6 ->
4x SR network -> PIL LANCZOS resize back) but with onnxruntime instead of torch.
Float32 rounding differs slightly from torch, which is exactly the noise floor the
golden fixtures are meant to tolerate.

  F:\gpt-anime-denoise\.venv\Scripts\python.exe android\tools\bench_models.py
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
TAGS = ["01ee25d8", "061d63a2", "4dde2cb0"]
SR = ["realesrgan_x4plus_anime_6b", "realesr_animevideov3_x4"]


def make_session(name):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 0
    return ort.InferenceSession(str(MODELS / f"{name}.onnx"), opts, providers=["CPUExecutionProvider"])


def run_sr(sess, small):
    """small (RGB PIL) -> 4x RGB PIL, in the same way upscale() would."""
    x = (np.asarray(small, np.float32) / 255).transpose(2, 0, 1)[None]
    y = sess.run(None, {"input": x})[0]
    y = np.clip(y, 0, 1)[0].transpose(1, 2, 0)
    return Image.fromarray((y * 255 + 0.5).astype(np.uint8))


def pipeline(sess, src):
    W, H = src.size
    t = time.perf_counter()
    small = src.resize((round(W * dn.DOWN), round(H * dn.DOWN)), Image.BOX)
    t_down = time.perf_counter() - t
    t = time.perf_counter()
    big = run_sr(sess, small)
    t_sr = time.perf_counter() - t
    t = time.perf_counter()
    sr = big.resize((W, H), Image.LANCZOS)
    t_back = time.perf_counter() - t
    return sr, dict(downscale_s=t_down, sr_s=t_sr, resize_back_s=t_back)


def main():
    sessions = {n: make_session(n) for n in SR}
    for n, s in sessions.items():
        print(f"{n}: in={s.get_inputs()[0].shape} out={s.get_outputs()[0].shape}", flush=True)

    # Warm-up: 3 runs at the real geometry so oneDNN primitives, the ORT thread
    # pool and the allocator arena are all established before anything is timed.
    warm = Image.fromarray(np.random.default_rng(0).integers(0, 256, (307, 307, 3), dtype=np.uint8))
    for _ in range(3):
        for s in sessions.values():
            run_sr(s, warm)

    report = {}
    mids = {}
    for name, sess in sessions.items():
        entry = {"model_mb": round((MODELS / f"{name}.onnx").stat().st_size / 1e6, 3), "runs": []}
        for tag in TAGS:
            src = dn.open_image(GOLDEN / f"in_{tag}.png")[0]
            runs = []
            for _ in range(3):                      # best-of-3: this box also runs ComfyUI
                sr, times = pipeline(sess, src)
                runs.append(times)
            times = {k: min(r[k] for r in runs) for k in runs[0]}
            mid = dn.to_arr(sr)
            mids[(name, tag)] = mid
            entry["runs"].append(dict(tag=tag, **{k: round(v, 4) for k, v in times.items()},
                                      sr_s_all=[round(r["sr_s"], 3) for r in runs]))
            print(f"  {name} {tag}: down {times['downscale_s']*1000:.1f} ms | sr {times['sr_s']:.2f} s "
                  f"| back {times['resize_back_s']*1000:.1f} ms", flush=True)
        # median SR time over the three inputs
        entry["sr_s_median"] = round(sorted(r["sr_s"] for r in entry["runs"])[1], 3)
        # full-pipeline result for the L0 timing comparison
        entry["l0_s"] = None
        report[name] = entry

    # Quality: reference = golden mid_*.npy (produced by torch + denoise.py exactly).
    for tag in TAGS:
        ref = np.load(GOLDEN / f"mid_{tag}.npy")
        for name in SR:
            d = np.abs(mids[(name, tag)] - ref)
            report[name].setdefault("quality_vs_torch_ref", {})[tag] = dict(
                max_abs=float(d.max()), mean_abs=float(d.mean()),
                max_levels=float(d.max() * 255), mean_levels=float(d.mean() * 255))
            print(f"  quality {name} {tag}: max {d.max():.6f} ({d.max()*255:.2f}/255) "
                  f"mean {d.mean():.7f} ({d.mean()*255:.4f}/255)", flush=True)

    # 6B vs v3 head-to-head (how much does the small model change the picture?)
    for tag in TAGS:
        d = np.abs(mids[("realesr_animevideov3_x4", tag)] - mids[("realesrgan_x4plus_anime_6b", tag)])
        report.setdefault("v3_vs_6b", {})[tag] = dict(max_abs=float(d.max()), mean_abs=float(d.mean()))
        print(f"  v3 vs 6B {tag}: max {d.max():.6f} mean {d.mean():.6f} ({d.mean()*255:.3f}/255)", flush=True)

    # L0 stage timing on the ORT middle results (same cost either way).
    for tag in TAGS:
        t = time.perf_counter()
        dn.l0_smooth(mids[("realesr_animevideov3_x4", tag)], dn.L0, device="cpu")
        print(f"  l0 on {tag}: {(time.perf_counter()-t)*1000:.0f} ms", flush=True)

    out = HERE / "model_benchmark.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


main()
