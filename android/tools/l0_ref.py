#!/usr/bin/env python
"""Standalone reference implementation of the L0 gradient-minimisation stage.

This is a byte-for-byte port of the CPU branch of denoise.l0_smooth
(Xu et al. 2011, "Image Smoothing via L0 Gradient Minimization"). It has NO
torch dependency - only numpy and Pillow - so it runs anywhere and can be used
as an oracle when porting the stage to Kotlin.

Input : an HxWx3 .npy array of floats in [0, 1] (float32 or float64; float32
        is promoted to float64, exactly like denoise.l0_smooth does).
Output: <stem>_l0.npy (float64 HxWx3, clipped to [0, 1]) and
        <stem>_l0.png (8-bit, quantised as (a * 255 + 0.5).astype(uint8),
        the same as denoise.to_img).

Usage:
    python l0_ref.py input.npy [-o OUTPUT_STEM] [--lam 0.001] [--kappa 2.0] [--beta-max 1e5]

Defaults match denoise.L0 = 0.001 / kappa 2.0 / beta_max 1e5.

Numeric notes for the Kotlin port
---------------------------------
* The whole stage is float64. FFTs are unnormalised numpy.fft.fft2 / ifft2
  evaluated over axes (0, 1) of an HxWx3 array, i.e. one 2-D FFT per colour
  channel.
* fx/fy are the forward-difference kernels with a single -1 at [0,0] and a +1
  at [0, W-1] / [H-1, 0]: that is a CIRCULANT (wrap-around) difference, not a
  symmetric one. Periodicity is global, so the result genuinely depends on the
  whole image and cannot be computed independently per tile.
* S is re-derived from N1 (the FFT of the INPUT) on every iteration; the target
  N1 is never updated, only div is.
* Loop runs while beta < beta_max with beta starting at 2 * lam and multiplied by
  kappa each pass: 26 iterations for the defaults
  (2e-3 * 2^25 = 6.7e4 < 1e5 <= 2e-3 * 2^26 = 1.34e5).
* The comparison is strict (<), and the band test is a strict < as well; both
  matter if you reimplement with float32 accumulation.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def l0_smooth(img, lam, kappa=2.0, beta_max=1e5):
    """L0 gradient minimisation on an HxWx3 float image in [0, 1] (denoise.py CPU path)."""
    S = np.asarray(img).astype(np.float64)
    H, W, _ = S.shape
    fx = np.zeros((H, W)); fx[0, 0], fx[0, -1] = -1, 1
    fy = np.zeros((H, W)); fy[0, 0], fy[-1, 0] = -1, 1
    otf = np.abs(np.fft.fft2(fx)) ** 2 + np.abs(np.fft.fft2(fy)) ** 2
    N1 = np.fft.fft2(S, axes=(0, 1))
    beta = 2 * lam
    while beta < beta_max:
        h = np.roll(S, -1, 1) - S
        v = np.roll(S, -1, 0) - S
        t = (h ** 2 + v ** 2).sum(2) < lam / beta
        h[t] = 0
        v[t] = 0
        div = (np.roll(h, 1, 1) - h) + (np.roll(v, 1, 0) - v)
        S = np.real(np.fft.ifft2((N1 + beta * np.fft.fft2(div, axes=(0, 1))) / (1 + beta * otf)[..., None], axes=(0, 1)))
        beta *= kappa
    return S.clip(0, 1)


def to_img(a):
    """denoise.to_img: float [0,1] -> 8-bit, round-half-up."""
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="HxWx3 .npy, float32/float64 in [0, 1]")
    ap.add_argument("-o", "--out", help="output stem (default: <input stem>_l0)")
    ap.add_argument("--lam", type=float, default=0.001, help="L0 weight (denoise.L0, default 0.001)")
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--beta-max", type=float, default=1e5)
    a = ap.parse_args()

    src = Path(a.input)
    stem = Path(a.out) if a.out else src.with_name(src.stem + "_l0")
    arr = np.load(src)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise SystemExit(f"expected HxWx3, got {arr.shape}")

    out = l0_smooth(arr, a.lam, a.kappa, a.beta_max)
    np.save(str(stem) + ".npy", out)
    im = to_img(out)
    im.save(str(stem) + ".png", "PNG")
    print(f"{src} {arr.shape} {arr.dtype} -> {stem}.npy ({out.dtype}) + {stem}.png")
    print(f"  lam={a.lam:g} kappa={a.kappa:g} beta_max={a.beta_max:g} "
          f"iters={int(np.ceil(np.log(a.beta_max / (2 * a.lam)) / np.log(a.kappa)))}")
    print(f"  in  min/max/mean = {arr.min():.5f} {arr.max():.5f} {arr.mean():.5f}")
    print(f"  out min/max/mean = {out.min():.5f} {out.max():.5f} {out.mean():.5f}")


if __name__ == "__main__":
    main()
