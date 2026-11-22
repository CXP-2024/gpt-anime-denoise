import sys, time
sys.path.insert(0, r"F:\gpt-anime-denoise")
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort
from pathlib import Path

golden = Path(r"F:\gpt-anime-denoise\android\tools\golden")
out = Path(r"F:\gpt-anime-denoise\android\tools\quality")
out.mkdir(parents=True, exist_ok=True)
M6 = r"F:\gpt-anime-denoise\android\app\src\main\assets\realesrgan_x4plus_anime_6b.onnx"
MV3 = r"F:\gpt-anime-denoise\android\app\src\main\assets\realesr_animevideov3_x4.onnx"

def load(p): return ort.InferenceSession(p, providers=["CPUExecutionProvider"])
s6, s3 = load(M6), load(MV3)

def sr(sess, im):
    x = np.asarray(im, np.float32)/255.0
    x = np.ascontiguousarray(x.transpose(2,0,1))[None]
    y = sess.run(None, {"input": x})[0]
    return Image.fromarray(np.clip((y[0].transpose(1,2,0)*255.0+0.5).astype(np.int32),0,255).astype(np.uint8))

for tag in ["01ee25d8", "061d63a2"]:
    src = Image.open(golden / f"in_{tag}.png").convert("RGB")
    W, H = src.size
    small = src.resize((round(W*0.6), round(H*0.6)), Image.BOX)
    t0 = time.time(); up6 = sr(s6, small); t6 = time.time()-t0
    t0 = time.time(); up3 = sr(s3, small); t3 = time.time()-t0
    b6 = up6.resize((W, H), Image.LANCZOS)
    b3 = up3.resize((W, H), Image.LANCZOS)
    d = np.abs(np.asarray(b6, np.int32) - np.asarray(b3, np.int32))
    print(f"{tag}: 6B {t6:.1f}s vs v3 {t3:.1f}s = {t6/max(t3,0.01):.1f}x",
          f"| max diff {d.max()} mean {d.mean():.2f} | pixels differing {np.count_nonzero(d)/d.size*100:.1f}%")
    # side by side, plus a 3x nearest-neighbour crop of a detailed area
    sheet = Image.new("RGB", (W*2+16, H), (32,32,32))
    sheet.paste(b6, (0,0)); sheet.paste(b3, (W+16,0))
    dr = ImageDraw.Draw(sheet)
    dr.text((8,8), "6B (default)", fill=(255,255,0))
    dr.text((W+24,8), "animevideov3 (fast)", fill=(255,255,0))
    sheet.save(out / f"compare_{tag}.png")
    # zoom crop: find the highest-detail quarter and blow it up 3x without smoothing
    cx, cy = W//2, H//3
    box = (cx-80, cy-80, cx+80, cy+80)
    c6 = b6.crop(box).resize((480,480), Image.NEAREST)
    c3 = b3.crop(box).resize((480,480), Image.NEAREST)
    orig = src.crop(box).resize((480,480), Image.NEAREST)
    z = Image.new("RGB", (480*3+24, 480), (32,32,32))
    z.paste(orig,(0,0)); z.paste(c6,(480+12,0)); z.paste(c3,(960+24,0))
    dz = ImageDraw.Draw(z)
    dz.text((6,6), "original (noisy)", fill=(0,255,0))
    dz.text((480+18,6), "6B", fill=(255,255,0))
    dz.text((960+30,6), "v3", fill=(255,255,0))
    z.save(out / f"zoom_{tag}.png")
    print(f"  wrote compare_{tag}.png and zoom_{tag}.png")