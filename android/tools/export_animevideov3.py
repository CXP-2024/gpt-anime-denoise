
"""Export the optional lightweight realesr-animevideov3 model to ONNX.

Reuses export_onnx.export() verbatim (same opset, dynamic axes, numeric check) but
only for this model, so the already-verified 6B ONNX is never rewritten.

  F:\gpt-anime-denoise\.venv\Scripts\python.exe android\tools\export_animevideov3.py [outdir] [cache]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "models"
CACHE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"C:\Users\21294\.cache\gpt-anime-denoise")

sys.argv = ["export_onnx.py", str(OUT), str(CACHE)]   # export_onnx reads argv at import time
sys.path.insert(0, str(HERE))
import export_onnx as eo                                    # noqa: E402

pth = CACHE / "realesr-animevideov3.pth"
if not pth.exists():
    raise SystemExit(f"missing {pth}")
eo.export("realesr_animevideov3_x4", pth)
