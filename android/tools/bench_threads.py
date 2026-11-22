"""Thread-scaling check: how each model behaves on 1 thread vs all cores.

A phone has 1 big core + 3-5 little ones, so the 1-thread number is the closer
analogue of on-device cost than the 16-thread dev-box number.

  .venv\\Scripts\\python.exe android\\tools\\bench_threads.py
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
NAMES = ["realesrgan_x4plus_anime_6b", "realesr_animevideov3_x4"]
SIZE = 307  # what a 512x512 image becomes at DOWN=0.6


def bench(name, threads):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.intra_op_num_threads = threads
    s = ort.InferenceSession(str(MODELS / (name + ".onnx")), o, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((1, 3, SIZE, SIZE), dtype=np.float32)
    for _ in range(2):
        s.run(None, {"input": x})
    times = []
    for _ in range(3):
        t = time.perf_counter()
        s.run(None, {"input": x})
        times.append(time.perf_counter() - t)
    return min(times)


res = {}
print(str(SIZE) + "x" + str(SIZE) + " input -> " + str(SIZE * 4) + "x" + str(SIZE * 4) + " output")
for name in NAMES:
    r = {}
    for label, th in (("1 thread", 1), ("4 threads", 4), ("all cores", 0)):
        r[label] = round(bench(name, th), 3)
    res[name] = r
    print("  %-30s " % name + " | ".join("%s %.2f s" % (k, v) for k, v in r.items()))
(HERE / "thread_benchmark.json").write_text(json.dumps(res, indent=2) + "\n")
