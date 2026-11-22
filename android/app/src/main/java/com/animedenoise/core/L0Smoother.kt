package com.animedenoise.core

/**
 * L0 gradient minimisation (Xu, Lu, Xie & Yang, "Image Smoothing via L0 Gradient
 * Minimization", SIGGRAPH Asia 2011) - the last step of the denoise pipeline.
 *
 * This is a faithful port of `denoise.l0_smooth`, the float64 CPU reference:
 *
 * ```python
 * S = img.astype(np.float64)                 # H x W x 3, values in [0, 1]
 * fx = np.zeros((H, W)); fx[0, 0], fx[0, -1] = -1, 1
 * fy = np.zeros((H, W)); fy[0, 0], fy[-1, 0] = -1, 1
 * otf = np.abs(np.fft.fft2(fx)) ** 2 + np.abs(np.fft.fft2(fy)) ** 2
 * N1 = np.fft.fft2(S, axes=(0, 1))
 * beta = 2 * lam
 * while beta < beta_max:
 *     h = np.roll(S, -1, 1) - S
 *     v = np.roll(S, -1, 0) - S
 *     t = (h ** 2 + v ** 2).sum(2) < lam / beta
 *     h[t] = 0
 *     v[t] = 0
 *     div = (np.roll(h, 1, 1) - h) + (np.roll(v, 1, 0) - v)
 *     S = np.real(np.fft.ifft2((N1 + beta * np.fft.fft2(div, axes=(0, 1)))
 *                              / (1 + beta * otf)[..., None], axes=(0, 1)))
 *     beta *= kappa
 * return S.clip(0, 1)
 * ```
 *
 * What the port adds is speed, not different mathematics:
 *
 *  * `N1` (the spectrum of the source) and `otf` are computed once, exactly as above;
 *  * the three colour planes travel through the FFT together, so the row and column loops
 *    hold three times as many independent items and keep the pool busy;
 *  * `1 / (1 + beta * otf)` is precomputed once per iteration instead of once per plane;
 *  * the divergence overwrites the `h` buffer, holding the footprint at eight H x W planes
 *    instead of nine (about 96 MB for 1024x1536 in double precision);
 *  * the loop count is identical: `beta` starts at `2 * lam` and is multiplied by `kappa`
 *    while it stays below `betaMax` - 22 iterations at the default `lam = 0.001`.
 *
 * Semantics that are easy to get wrong, spelled out:
 *
 *  * `h` and `v` are forward differences along x and y with circular wrap-around.
 *  * `t` is a *per-pixel* mask summed over all three planes. Zeroing an element of the
 *    H x W x 3 array zeroes that pixel in every channel, so the decision has to be taken
 *    once for the whole pixel - that is the fused mask pass in [smooth].
 *  * `np.fft.ifft2` carries a `1 / (H * W)` scale; [Fft2D] applies the same one.
 *  * every intermediate is float64, matching the reference's `astype(np.float64)`.
 *
 * Typical strength is 0.0005 (gentle) to 0.002 (strong); 0.001 is the shipped default.
 * Larger values flatten more, and past about 0.01 soft blushes and gradients disappear.
 */
object L0Smoother {

    /**
     * Smooths [img] with L0 gradient minimisation and returns a new image.
     *
     * [img] is not modified. Every returned sample is clamped to `[0, 1]`, which is what
     * the reference's `S.clip(0, 1)` does.
     *
     * `param img         source image, nominally in `[0, 1]`
     * `param lam         `lambda`, the L0 strength; must be positive. 0.001 is the
     *                    reference default and 0.0005..0.002 the useful range.
     * `param kappa       geometric growth of the penalty weight; 2.0 in the reference
     * `param betaMax     the loop ends once `beta >= betaMax`; 1e5 in the reference
     * `param isCancelled polled once per iteration. Returning `true` throws
     *                    [L0CancelledException] - one iteration of a large image takes
     *                    hundreds of milliseconds, so this is how a UI stop button works.
     * `throws IllegalArgumentException if an argument is out of range
     * `throws L0CancelledException when [isCancelled] asks to stop
     */
    fun smooth(
        img: FloatImage,
        lam: Double,
        kappa: Double = 2.0,
        betaMax: Double = 1e5,
        isCancelled: () -> Boolean = { false }
    ): FloatImage {
        require(lam > 0.0) { "lam must be > 0, was " + lam }
        require(kappa > 1.0) { "kappa must be > 1, was " + kappa }
        require(betaMax > 0.0) { "betaMax must be > 0, was " + betaMax }

        val w = img.width
        val hgt = img.height
        val n = w * hgt
        val fft = Fft2D(hgt, w)

        // N1 = fft2(S), one spectrum per colour plane, kept for the whole loop.
        val n1Re = planes(n)
        val n1Im = planes(n)
        for (c in 0 until 3) {
            val src = img.plane(c)
            val dst = n1Re[c]
            for (i in 0 until n) dst[i] = src[i].toDouble()
        }
        fft.transformChannels(n1Re, n1Im, false)

        val otf = computeOtf(fft, hgt, w)
        val invOtf = DoubleArray(n)

        // S in the spatial domain, plus the difference and divergence buffers.
        var s = planes(n)
        for (c in 0 until 3) {
            val src = img.plane(c)
            val dst = s[c]
            for (i in 0 until n) dst[i] = src[i].toDouble()
        }
        val hDiff = planes(n)
        val vDiff = planes(n)
        // Reused each iteration: the divergence, then its spectrum, then the new S.
        var workRe = planes(n)
        val workIm = planes(n)

        var beta = 2.0 * lam
        while (beta < betaMax) {
            if (isCancelled()) throw L0CancelledException()
            val threshold = lam / beta

            for (i in 0 until n) invOtf[i] = 1.0 / (1.0 + beta * otf[i])

            // h = roll(S, -1, 1) - S and v = roll(S, -1, 0) - S, for the three planes.
            Parallel.forEach(3 * hgt) { idx ->
                val c = idx / hgt
                val y = idx % hgt
                forwardDiffRow(s[c], hDiff[c], y, w)
                forwardDiffColumn(s[c], vDiff[c], y, w, hgt)
            }

            // t = (h^2 + v^2).sum(2) < lam / beta, then h[t] = v[t] = 0 in every plane.
            Parallel.forEach(hgt) { y ->
                maskRow(hDiff, vDiff, y, w, threshold)
            }

            // div = (roll(h, 1, 1) - h) + (roll(v, 1, 0) - v), written over hDiff.
            Parallel.forEach(3 * hgt) { idx ->
                val c = idx / hgt
                val y = idx % hgt
                divergenceRow(hDiff[c], vDiff[c], workRe[c], workIm[c], y, w, hgt)
            }

            // S = real(ifft2((N1 + beta * fft2(div)) / (1 + beta * otf)))
            fft.transformChannels(workRe, workIm, false)
            Parallel.forEach(3 * n) { idx ->
                val c = idx / n
                val i = idx % n
                val k = invOtf[i]
                workRe[c][i] = (n1Re[c][i] + beta * workRe[c][i]) * k
                workIm[c][i] = (n1Im[c][i] + beta * workIm[c][i]) * k
            }
            fft.transformChannels(workRe, workIm, true)

            // workRe now holds the new S; the old S planes become the next divergence target.
            val recycled = s
            s = workRe
            workRe = recycled
            beta *= kappa
        }

        val out = FloatImage(w, hgt)
        for (c in 0 until 3) {
            val dst = out.plane(c)
            val src = s[c]
            for (i in 0 until n) dst[i] = clamp01(src[i]).toFloat()
        }
        return out
    }

    private fun planes(n: Int): Array<DoubleArray> =
        arrayOf(DoubleArray(n), DoubleArray(n), DoubleArray(n))

    /** `|fft2(fx)|^2 + |fft2(fy)|^2`; both kernels are real, so the imaginary part is dropped. */
    private fun computeOtf(fft: Fft2D, hgt: Int, w: Int): DoubleArray {
        val n = hgt * w
        val fx = DoubleArray(n)
        val fxi = DoubleArray(n)
        fx[0] = -1.0
        fx[w - 1] = 1.0
        fft.forward(fx, fxi)
        val otf = DoubleArray(n)
        for (i in 0 until n) otf[i] = fx[i] * fx[i] + fxi[i] * fxi[i]

        val fy = DoubleArray(n)
        val fyi = DoubleArray(n)
        fy[0] = -1.0
        fy[(hgt - 1) * w] = 1.0
        fft.forward(fy, fyi)
        for (i in 0 until n) otf[i] += fy[i] * fy[i] + fyi[i] * fyi[i]
        return otf
    }

    /** `h[y][x] = S[y][x + 1] - S[y][x]`, with `x + 1` wrapping around to 0. */
    private fun forwardDiffRow(s: DoubleArray, dst: DoubleArray, y: Int, w: Int) {
        val base = y * w
        val last = base + w - 1
        for (x in base until last) dst[x] = s[x + 1] - s[x]
        dst[last] = s[base] - s[last]
    }

    /** `v[y][x] = S[y + 1][x] - S[y][x]`, with `y + 1` wrapping around to 0. */
    private fun forwardDiffColumn(s: DoubleArray, dst: DoubleArray, y: Int, w: Int, hgt: Int) {
        val base = y * w
        val next = if (y == hgt - 1) 0 else base + w
        for (x in 0 until w) dst[base + x] = s[next + x] - s[base + x]
    }

    /**
     * Applies the joint L0 mask to row [y].
     *
     * `h` and `v` are real here (they only become complex after the transform), so the
     * magnitude the reference sums is just `h*h + v*v`. The three planes are visited at
     * once because one pixel below the threshold is cleared in every plane.
     */
    private fun maskRow(hDiff: Array<DoubleArray>, vDiff: Array<DoubleArray>, y: Int, w: Int, threshold: Double) {
        val h0 = hDiff[0]
        val h1 = hDiff[1]
        val h2 = hDiff[2]
        val v0 = vDiff[0]
        val v1 = vDiff[1]
        val v2 = vDiff[2]
        val base = y * w
        for (x in 0 until w) {
            val i = base + x
            val a0 = h0[i]
            val b0 = v0[i]
            val a1 = h1[i]
            val b1 = v1[i]
            val a2 = h2[i]
            val b2 = v2[i]
            if (a0 * a0 + b0 * b0 + a1 * a1 + b1 * b1 + a2 * a2 + b2 * b2 < threshold) {
                h0[i] = 0.0
                v0[i] = 0.0
                h1[i] = 0.0
                v1[i] = 0.0
                h2[i] = 0.0
                v2[i] = 0.0
            }
        }
    }

    /**
     * `div[y][x] = (h[y][x-1] - h[y][x]) + (v[y-1][x] - v[y][x])` for one plane and row,
     * written into [dst] with its imaginary part cleared.
     */
    private fun divergenceRow(
        h: DoubleArray,
        v: DoubleArray,
        dst: DoubleArray,
        dstIm: DoubleArray,
        y: Int,
        w: Int,
        hgt: Int
    ) {
        val base = y * w
        val prevRow = if (y == 0) (hgt - 1) * w else base - w
        for (x in 0 until w) {
            val i = base + x
            val hPrev = if (x == 0) h[base + w - 1] else h[i - 1]
            dst[i] = (hPrev - h[i]) + (v[prevRow + x] - v[i])
            dstIm[i] = 0.0
        }
    }

    private fun clamp01(v: Double): Double = if (v < 0.0) 0.0 else if (v > 1.0) 1.0 else v
}

/** Thrown by [L0Smoother.smooth] when its `isCancelled` callback returns `true`. */
class L0CancelledException : RuntimeException("L0 smoothing cancelled")
