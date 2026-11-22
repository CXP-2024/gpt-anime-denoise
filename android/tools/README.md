# Android port - build-time tools, models and golden fixtures

Build-time only: nothing here ships in the APK except the `.onnx` files, which
are copied to `..\app\src\main\assets\`.

## Models shipped in the APK

| assets file | size | role |
|---|---|---|
| `realesrgan_x4plus_anime_6b.onnx` | 17,961,298 B (17.13 MiB) | **default / quality tier** |
| `realesr_animevideov3_x4.onnx` | 2,492,874 B (2.38 MiB) | fast tier (optional) |

Both: opset 17, `input` `[1,3,H,W]` dynamic, `output` `[1,3,4h,4w]` dynamic,
NCHW float32 in [0,1], `onnx.checker` clean. Exported by `export_onnx.py`
(6B) and `export_animevideov3.py` (compact), from the same spandrel/torch
code path `denoise.py` uses.

Numeric fidelity: 6B ONNX vs torch max abs diff 4e-6 / 3e-6 on random inputs;
compact ONNX vs torch max abs diff 3e-6 / 5e-6. Both are pure export rounding.

## Golden fixtures (`golden\`)

Three 512x512 crops of real outputs, run through the exact `denoise.py`
default pipeline on CPU. See `golden\README.md` for the file-by-file
description and the tolerances. Short version:

| stage | fixture | pass criterion |
|---|---|---|
| super-resolution | `mid_<tag>.npy` (float64 NxNx3) | max abs <= 0.0040 (1 level), mean <= 5e-6 |
| L0 | `out_<tag>.png` | max <= 4 levels, mean <= 0.05 levels, <=1% of pixels differing |
| end to end | `out_<tag>.png` | max <= 4 levels, <=1% of pixels differing |

The compact tier has its **own** references, `golden\v3_mid_*.npy` and
`golden\v3_out_*.png`, and must be tested against those:

| stage | fixture | pass criterion |
|---|---|---|
| v3 super-resolution | `v3_mid_<tag>.npy` | replay is exact (max 0); use the 6B gate with room to spare |
| v3 L0 / end to end | `v3_out_<tag>.png` | as above |

**v3 vs the 6B goldens is not a tolerance question.** Measured difference:
148-165 levels on edge pixels, 3.3-4.1 levels mean, 89-91% of pixels differ -
it fails the 6B gate outright. The compact network renders a *different, softer*
picture. Never point the v3 tier at `mid_*` / `out_*`.

## Scripts

| script | purpose |
|---|---|
| `export_onnx.py` | exports `RealESRGAN_x4plus_anime_6B.pth` (and `realesr-animevideov3.pth` if present) to ONNX, then verifies with onnxruntime. |
| `export_animevideov3.py` | exports only the compact model, leaving the verified 6B ONNX untouched. |
| `generate_fixtures.py` | builds the 6B fixtures in `golden\` from three real images with the exact `denoise.py` default pipeline (CPU, ~55 s). |
| `generate_v3_fixtures.py` | builds the compact-model fixtures (`v3_mid_*`, `v3_out_*`) with the same pipeline and the v3 network swapped in. |
| `l0_ref.py` | torch-free standalone L0 reference (numpy + Pillow). **Bit-identical** to `denoise.l0_smooth` on all three fixtures. |
| `bench_models.py` | per-fixture stage timing + quality of both ONNX models vs the golden fixtures. Writes `model_benchmark.json`. |
| `bench_forward.py` | one 512x512 -> 2048x2048 forward pass, best-of-N. Writes `forward_benchmark.json`. |
| `bench_threads.py` | thread scaling (1 / 4 / all cores) at 307x307 -> 1228x1228. Writes `thread_benchmark.json`. |
| `time_models.py` | interleaved best-of-5 timing of the two models, full pipeline. Writes `timing_models.json`. |
| `model_cost.py` | parameter count and MACs/px from a symbolic trace of each graph. Writes `model_cost.json`. |

`model_compare\` holds side-by-side and amplified-difference images for the two
models (see below).

## Measured CPU cost (dev box: Intel Core Ultra 5 125H, onnxruntime 1.23.2, CPU EP)

The pipeline geometry for a 512x512 image: 512x512 -> BOX downscale to 307x307
-> model -> 1228x1228 -> LANCZOS back to 512x512.

Per stage, best of 5, models interleaved in one process. The two resize rows are
the *same* PIL call for both models - the 3x spread there is dev-box noise on a
sub-100 ms operation, not a model property.

| stage | 6B | compact v3 |
|---|---|---|
| downscale (PIL BOX) | 12.6 ms | 12.6 ms |
| **SR inference** | **7.64 s** | **1.77 s** |
| resize back (PIL LANCZOS) | 61 ms | 21 ms |
| L0 (float64 numpy, 26 iters) | 6.80 s | 6.80 s |
| **total** | **14.52 s** | **8.61 s** |

Isolated 512x512 -> 2048x2048 forward pass (`forward_benchmark.json`):
6B best 26.2 s / median 28.3 s; v3 best 2.48 s / median 2.68 s; **v3 10.6x faster**.

Thread scaling at 307x307 -> 1228x1228 (`thread_benchmark.json`):

| model | 1 thread | 4 threads | all cores |
|---|---|---|---|
| 6B | 29.35 s | 9.19 s | 10.11 s |
| v3 | 2.18 s | 0.97 s | 0.89 s |

Static cost (`model_cost.json`), from a symbolic trace of each graph:

| model | params | convs | MACs/px | FLOPs/px |
|---|---|---|---|---|
| 6B | 4.468 M | 96 | 5.71 M | 11.41 M |
| v3 | 0.621 M | 18 | 0.62 M | 1.24 M |

6B = 9.2x the compute, 7.2x the parameters.
A 512x512 image costs 538 GMAC (1.08 TFLOP) on 6B and 58 GMAC (117 GFLOP) on v3.

## Quality of the compact model

Against the 6B result on the same golden inputs, after the identical
downscale + resize-back pipeline (so this is purely "does the smaller network
render the same picture"):

| fixture | PSNR | mean abs diff |
|---|---|---|
| `01ee25d8` | 32.24 dB | 3.53/255 |
| `061d63a2` | 32.20 dB | 3.46/255 |
| `4dde2cb0` | 30.23 dB | 4.39/255 |

Mean difference 3.5-4.4 levels sounds small, but the amplified difference maps
in `model_compare\diff_*.png` show it is concentrated exactly on **line work and
hair strands** - the compact model draws them softer and slightly displaced
(max diff 146-166 levels on individual edge pixels). Side by side
(`model_compare\cmp_*.png`: input | 6B | v3) 6B keeps the sharp ink lines; v3
visibly softens them.

## Recommendation

**Ship 6B as the default; offer v3 only as an explicit "fast" toggle.**

* The stage that dominates on-device cost is the network, and 6B is 9.2x the
  compute of v3. On the dev CPU that is 4.3x wall-clock (7.6 s vs 1.8 s) at 307x307.
  Extrapolating from the 1-thread numbers (an Adreno 725's ALU throughput per
  thread is well below a Lion Cove core, and GPU convs need 3-5x more threads to
  saturate), 6B should be expected in the tens of seconds per 512x512 image if it
  ends up on CPU, and low single digits only on the GPU path. **This is an
  extrapolation - measure it on the device before committing to a default.**
* **Route 6B to the GPU (NNAPI, then XNNPACK).** It is a plain 96-conv/2-Resize
  graph with no exotic ops; both `Resize` (nearest, 2x) and `LeakyRelu` are
  NNAPI-supported, so the whole graph should stay on the accelerator. Falling
  back to CPU for 6B is the case to avoid - that is where a 30-60 s wait lives.
* **v3 is the safe CPU fallback**, not a default: 117 GFLOP is realistic on the
  little cores, and on the 1-thread number (2.18 s vs 29.35 s) it is 13.5x
  cheaper, so it degrades far more gracefully.
* **v3 does not reproduce the desktop result.** Against the 6B goldens it is off
  by 148-165 levels on edges and ~3.3-4.1 levels mean (89-91% of pixels). It
  exports, runs, and validates against its own reference exactly - it is a
  working model, just a different look. If users expect "same output as
  denoise.py", the fast tier must be labelled as a quality trade, not as a
  speed-only switch.
* **Do not skip the L0 stage for speed on the GPU path**: L0 is 47% of the 6B total and 79% of the v3 total on CPU, but it is FFT-bound and will *also* be slow on the phone. Do it in float64 as `denoise.py` does - a float32 FFT measured 7e-4..1.4e-2 absolute error against the reference, up to 3 levels on dark gradients.

## Regenerating everything

```
cd F:\gpt-anime-denoise
set USERPROFILE=C:\Users\21294
.venv\Scripts\python.exe android\tools\generate_fixtures.py
.venv\Scripts\python.exe android\tools\export_onnx.py android\tools\models
.venv\Scripts\python.exe android\tools\export_animevideov3.py android\tools\models
.venv\Scripts\python.exe android\tools\bench_models.py
```

`realesr-animevideov3.pth` (2,504,012 B, sha256
`b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d`) was fetched
from `https://hf-mirror.com/Mitroshenkov87/voxprint-mirror-realesr-animevideov3`
because github.com and the ghproxy/ghfast mirrors were unreachable from this
host. `export_onnx.py` looks for it in `C:\Users\21294\.cache\gpt-anime-denoise`.
