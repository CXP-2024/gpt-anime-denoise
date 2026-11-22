"""Benchmark a single 512x512 -> 2048x2048 forward pass with onnxruntime (CPU).

The numbers the Android port needs to reason about: one model inference at the
geometry the default pipeline actually runs.

  .venv\Scripts\python.exe android\tools\bench_forward.py [reps]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
NAMES = ["realesrgan_x4plus_anime_6b", "realesr_animevideov3_x4"]
SIZE = 512
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def session(name, threads=None):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        o.intra_op_num_threads = threads
    return ort.InferenceSession(str(MODELS / f"{name}.onnx"), o, providers=["CPUExecutionProvider"])


def main():
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, SIZE, SIZE), dtype=np.float32)
    out = {}
    print(f"forward pass {SIZE}x{SIZE} -> {SIZE*4}x{SIZE*4}, onnxruntime {ort.__version__}, CPU EP")

    for name in NAMES:
        sess = session(name)
        for _ in range(2):                       # warm-up: oneDNN primitives, thread pool, arena
            sess.run(None, {"input": x})
        times = []
        for _ in range(REPS):
            t = time.perf_counter()
            y = sess.run(None, {"input": x})[0]
            times.append(time.perf_counter() - t)
        mb = (MODELS / f"{name}.onnx").stat().st_size / 1e6
        best, med = min(times), sorted(times)[len(times) // 2]
        out[name] = dict(model_mb=round(mb, 3), out_shape=list(y.shape),
                         best_s=round(best, 3), median_s=round(med, 3),
                         all_s=[round(t, 3) for t in times])
        print(f"  {name:30s} {mb:6.2f} MB | out {y.shape} | best {best:.2f} s | median {med:.2f} s | all {[round(t,2) for t in times]}")
    a = out[NAMES[0]]["best_s"]
    b = out[NAMES[1]]["best_s"]
    out["speedup_v3_over_6b"] = round(a / b, 2)
    print(f"  v3 is {a/b:.2f}x faster on CPU at this geometry")

    (HERE / "forward_benchmark.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", HERE / "forward_benchmark.json")


main()
