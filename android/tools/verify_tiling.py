
"""Verify that horizontal band tiling with a 100-row halo is bit-identical to a whole-image pass,
using the exact same preprocess/postprocess rounding the Kotlin code uses."""
import numpy as np, onnxruntime as ort
p = r"F:\gpt-anime-denoise\android\tools\models\realesrgan_x4plus_anime_6b.onnx"
sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
rng = np.random.default_rng(7)
W, H = 40, 300
img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)

def preprocess(y0, rows):
    band = img[y0:y0+rows]                       # rows x W x 3 uint8 (RGBA in app, RGB equivalent)
    x = band.astype(np.float32) / 255.0          # same as px * (1/255f) in float32
    return np.ascontiguousarray(x.transpose(2,0,1))[None]

def postprocess(out):
    # out: 1 x 3 x 4h x 4w float32 -> uint8 with (v*255+0.5) truncation
    return np.clip((out[0].transpose(1,2,0) * 255.0 + 0.5).astype(np.int32), 0, 255).astype(np.uint8)

whole = postprocess(sess.run(None, {"input": preprocess(0, H)})[0])
print("whole:", whole.shape)

HALO, BAND = 100, 192
def tiled():
    out = np.zeros((H*4, W*4, 3), np.uint8)
    r0 = 0
    while r0 < H:
        r1 = min(H, r0+BAND); a, b = max(0, r0-HALO), min(H, r1+HALO)
        y = sess.run(None, {"input": preprocess(a, b-a)})[0]
        px = postprocess(y)[ (r0-a)*4:(r1-a)*4 ]
        out[r0*4:r1*4] = px
        r0 = r1
    return out

t = tiled()
d = np.abs(t.astype(int) - whole.astype(int))
print(f"tiled vs whole: max={d.max()} mean={d.mean():.6f} differing={np.count_nonzero(d)}")
for band in (32, 64, 160, 300):
    BAND_ = band
    out = np.zeros((H*4, W*4, 3), np.uint8); r0=0
    while r0 < H:
        r1 = min(H, r0+band); a, b = max(0, r0-HALO), min(H, r1+HALO)
        y = sess.run(None, {"input": preprocess(a, b-a)})[0]
        out[r0*4:r1*4] = postprocess(y)[(r0-a)*4:(r1-a)*4]; r0 = r1
    dd = np.abs(out.astype(int) - whole.astype(int))
    print(f"band={band:4d} max={dd.max()} differing={np.count_nonzero(dd)}")
