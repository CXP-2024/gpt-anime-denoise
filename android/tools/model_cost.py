"""Static cost of the two ONNX models, from a symbolic shape trace of the graph.

Tracks (channels, spatial-multiplier) through every node, so the 4x upscale tail
is priced at 16x the pixels. Both nets are fully convolutional: MACs scale
linearly with input area, so macs_per_px is the single number that matters.
"""
import json
from pathlib import Path

import onnx
from onnx import numpy_helper

MODELS = Path(__file__).resolve().parent / "models"
NAMES = ["realesrgan_x4plus_anime_6b", "realesr_animevideov3_x4"]


def analyze(path):
    m = onnx.load(str(path))
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    nparam = sum(int(v.size) for v in inits.values())
    cin0 = int(g.input[0].type.tensor_type.shape.dim[1].dim_value)
    shp = {g.input[0].name: (cin0, 1)}          # name -> (channels, size multiplier)
    macs = 0.0
    convs = 0
    trace = []
    for node in g.node:
        src = shp.get(node.input[0]) if node.input else None
        if node.op_type == "Conv":
            w = inits[node.input[1]]
            cout, cin = int(w.shape[0]), int(w.shape[1])
            kh, kw = int(w.shape[-2]), int(w.shape[-1])
            groups = next((a.i for a in node.attribute if a.name == "group"), 1)
            s = src[1] if src else 1
            macs += (s ** 2) * cout * (cin // max(groups, 1)) * kh * kw
            convs += 1
            out = (cout, s)
        elif node.op_type == "DepthToSpace":
            b = next((a.i for a in node.attribute if a.name == "blocksize"), 1)
            out = (src[0] // (b * b), src[1] * b)
            trace.append(("DepthToSpace x%d" % b, out))
        elif node.op_type in ("Resize", "Upsample"):
            out = (src[0], src[1] * 2)
            trace.append(("%s x2" % node.op_type, out))
        else:
            out = src if src else (cin0, 1)
        for o in node.output:
            shp[o] = out
    return dict(params=int(nparam), convs=convs, out_scale=out[1],
                macs_per_px_m=round(macs / 1e6, 3),
                flops_per_px_m=round(macs * 2 / 1e6, 3),
                upscales=[t[0] for t in trace])


out = {}
for n in NAMES:
    r = analyze(MODELS / (n + ".onnx"))
    out[n] = r
    print("%-28s %6.3f M params | %3d convs | out x%d | %6.2f MMAC/px | %6.2f MFLOP/px | up: %s"
          % (n, r["params"] / 1e6, r["convs"], r["out_scale"], r["macs_per_px_m"],
             r["flops_per_px_m"], ", ".join(r["upscales"])))
print()
for n in NAMES:
    mpp = out[n]["macs_per_px_m"]
    print(n)
    for s, label in ((307, "512x512 image (DOWN=0.6)"), (614, "1024x1024 image"),
                     (1026, "1710x1710 image"), (1228, "2048x2048 image")):
        print("   %-26s input %4dx%-4d  %8.1f GMAC = %8.1f GFLOP"
              % (label, s, s, mpp * s * s / 1000, mpp * 2 * s * s / 1000))
    print()
ratio = out[NAMES[0]]["macs_per_px_m"] / out[NAMES[1]]["macs_per_px_m"]
pratio = out[NAMES[0]]["params"] / out[NAMES[1]]["params"]
print("6B is %.1fx the compute and %.1fx the parameters of the compact model" % (ratio, pratio))
out["ratio_6b_over_v3"] = dict(macs=round(ratio, 2), params=round(pratio, 2))
(Path(__file__).resolve().parent / "model_cost.json").write_text(json.dumps(out, indent=2) + "\n")
