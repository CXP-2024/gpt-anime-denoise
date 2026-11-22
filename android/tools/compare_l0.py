"""Compare two .npy arrays produced by the L0 parity harness.

Usage:
    python tools/compare_l0.py mine.npy reference.npy
    python tools/compare_l0.py --dir tools/kotlin-check/fixtures --suffix _out.npy --ref-suffix _ref.npy

Prints max absolute difference, mean absolute difference, RMS, PSNR (against a [0, 1] range,
which is what the pipeline produces) and the worst sample's coordinates. Exits non-zero when
the max absolute difference exceeds --tol (1e-6 by default), so it can gate CI.

Only numpy is required; no Pillow, no torch.
"""
import argparse
import os
import sys

import numpy as np

DEFAULT_TOL = 1e-6


def load(path):
    a = np.load(path)
    if a.ndim == 3 and a.shape[2] in (3, 4):
        return a[..., :3].astype(np.float64)
    return a.astype(np.float64)


def compare(mine_path, ref_path, tol=DEFAULT_TOL, quiet=False):
    """Return (ok, max_abs, mean_abs, psnr) for one pair of .npy files."""
    mine = load(mine_path)
    ref = load(ref_path)
    if mine.shape != ref.shape:
        print("%s: SHAPE MISMATCH %s vs %s" % (mine_path, mine.shape, ref.shape))
        return False, float("inf"), float("inf"), 0.0
    diff = np.abs(mine - ref)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    mse = float(np.mean(diff ** 2)) if diff.size else 0.0
    psnr = float("inf") if mse == 0.0 else 10.0 * np.log10(1.0 / mse)
    ok = max_abs <= tol
    if not quiet:
        print(os.path.basename(mine_path) + " vs " + os.path.basename(ref_path))
        print("  shape         = " + str(mine.shape))
        print("  max abs diff  = %.6e" % max_abs)
        print("  mean abs diff = %.6e" % mean_abs)
        print("  PSNR (range 1)= %.2f dB" % psnr)
        if diff.size:
            idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
            print("  worst sample  = " + str(tuple(int(i) for i in idx)) + "  mine=%.9f ref=%.9f" % (mine[idx], ref[idx]))
        print("  " + ("PASS" if ok else "FAIL") + " (tolerance %.1e)" % tol)
    return ok, max_abs, mean_abs, psnr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mine", nargs="?", help=".npy written by the Kotlin harness")
    ap.add_argument("ref", nargs="?", help=".npy of the Python float64 reference")
    ap.add_argument("--dir", help="compare every <tag><suffix> against <tag><ref-suffix>")
    ap.add_argument("--suffix", default="_out.npy", help="suffix of the Kotlin outputs")
    ap.add_argument("--ref-suffix", default="_ref.npy", help="suffix of the references")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL, help="max abs diff tolerance")
    a = ap.parse_args(argv)

    if a.dir:
        tags = sorted({f[:-len(a.suffix)] for f in os.listdir(a.dir) if f.endswith(a.suffix)})
        if not tags:
            print("no *%s files in %s" % (a.suffix, a.dir))
            return 2
        ok = True
        worst = (0.0, "")
        for tag in tags:
            mine = os.path.join(a.dir, tag + a.suffix)
            ref = os.path.join(a.dir, tag + a.ref_suffix)
            if not os.path.exists(ref):
                print(os.path.basename(mine) + ": no reference " + ref + " (skipped)")
                continue
            good, max_abs, _, _ = compare(mine, ref, a.tol)
            ok = ok and good
            if max_abs > worst[0]:
                worst = (max_abs, tag)
        print("")
        print("SUMMARY: %d fixtures, worst max abs diff %.6e (%s), tolerance %.1e -> %s"
              % (len(tags), worst[0], worst[1], a.tol, "PASS" if ok else "FAIL"))
        return 0 if ok else 1

    if not a.mine or not a.ref:
        ap.error("give two .npy paths, or --dir")
    ok, _, _, _ = compare(a.mine, a.ref, a.tol)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
