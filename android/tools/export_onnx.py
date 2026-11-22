
"""Export spandrel-compatible super-resolution .pth models to ONNX (fp32) for the Android app."""
import sys, os, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torch_runtime import import_torch  # isolate Intel DLLs like denoise.py does
torch = import_torch()
import spandrel

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)


def export(name: str, pth: Path, size=64):
    print(f"== {name} ==", flush=True)
    desc = spandrel.ModelLoader().load_from_file(str(pth))
    model = desc.model.eval()
    print("arch:", desc.architecture.name if hasattr(desc.architecture, "name") else desc.architecture,
          "scale:", desc.scale, flush=True)
    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        y = model(x)
    print("io:", x.shape, "->", y.shape, flush=True)
    onnx_path = OUT / f"{name}.onnx"
    torch.onnx.export(
        model, x, str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "H", 3: "W"}, "output": {2: "h", 3: "w"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print("wrote", onnx_path, onnx_path.stat().st_size, "bytes", flush=True)
    # numeric check with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        for s in (48, 96):
            xt = torch.rand(1, 3, s, s + 8 if s == 96 else s)
            with torch.no_grad():
                ref = model(xt).numpy()
            got = sess.run(None, {"input": xt.numpy()})[0]
            print(f"  size {tuple(xt.shape)}: max|diff|={np.abs(ref - got).max():.6f} "
                  f"mean|diff|={np.abs(ref - got).mean():.8f}", flush=True)
    except ImportError:
        print("  (onnxruntime not installed; skipped numeric check)", flush=True)


if __name__ == "__main__":
    cache = Path(sys.argv[2]) if len(sys.argv) > 2 else (Path(os.environ["USERPROFILE"]) / ".cache" / "gpt-anime-denoise")
    extra = cache
    export("realesrgan_x4plus_anime_6b", cache / "RealESRGAN_x4plus_anime_6B.pth")
    avv3 = extra / "realesr-animevideov3.pth"
    if avv3.exists():
        export("realesr_animevideov3_x4", avv3)
