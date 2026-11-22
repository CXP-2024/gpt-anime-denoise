import sys
sys.path.insert(0, r"F:\gpt-anime-denoise")
import numpy as np
from PIL import Image
import onnxruntime as ort
from pathlib import Path

g = Path(r"F:\gpt-anime-denoise\android\tools\golden")
src = Image.open(g / "in_01ee25d8.png").convert("RGB")
W, H = src.size
sw, sh = round(W*0.6), round(H*0.6)
print("src", src.size, "-> box", (sw, sh))

sess = ort.InferenceSession(r"F:\gpt-anime-denoise\android\tools\models\realesrgan_x4plus_anime_6b.onnx", providers=["CPUExecutionProvider"])
x = np.asarray(src.resize((sw, sh), Image.BOX), np.float32) / 255.0
x = np.ascontiguousarray(x.transpose(2,0,1))[None]
print("input tensor", x.shape, x.dtype, x.min(), x.max())
y = sess.run(None, {"input": x})[0]
print("output", y.shape, y.dtype, y.min(), y.max())
up = np.clip((y[0].transpose(1,2,0)*255.0+0.5).astype(np.int32), 0, 255).astype(np.uint8)

mid = np.load(g / "mid_01ee25d8.npy")
print("mid shape", mid.shape, "mean", float(mid.mean()))

back = Image.fromarray(up).resize((W, H), Image.LANCZOS)
back_arr = np.asarray(back, np.float64)/255
print("ORT-path SR->resize-back vs mid: max", float(np.abs(back_arr-mid).max()), "mean", float(np.abs(back_arr-mid).mean()))

import denoise as dn
from torch_runtime import import_torch
torch = import_torch()
model = dn.load_model("esrgan", Path(r"C:\Users\21294\.cache\gpt-anime-denoise"), "cpu")
ref_sr = dn.sr_at(model, src, "cpu")
ref = np.asarray(ref_sr.convert("RGB"), np.float64)/255
print("torch sr_at vs mid: max", float(np.abs(ref-mid).max()))

# compare the raw 4x tensor: ORT uint8 
big_ort = np.asarray(Image.fromarray(up), np.float32)/255
print("4x image mean: ort", float(big_ort.mean()), "vs torch", float(np.asarray(dn.upscale(model, src.resize((sw,sh), Image.BOX), "cpu"), np.float32).mean()/255))