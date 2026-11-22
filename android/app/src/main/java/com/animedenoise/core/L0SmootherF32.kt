package com.animedenoise.core

/**
 * Float32 L0 smoother for a single window that fits the heap.
 *
 * Used directly for small images and by [L0Tiled] for each tile of a large one. The whole solve,
 * FFT included, runs in single precision: measured drift against the exact [L0Smoother] is at most
 * 2 output levels, invisible at 8-bit, and it roughly halves both the memory and the FFT cost.
 *
 * Storage is Java-heap FloatArray. An off-heap variant was tried and measured slower; it is not
 * needed because [L0Tiled] keeps each solve bounded, so the heap never sees a megapixel-scale peak.
 */
object L0SmootherF32 {

    /** Convergence ceiling; measured visually identical to the reference's 1e5. */
    const val DEFAULT_BETA_MAX = 1e4

    fun smooth(
        img: FloatImage,
        lam: Double,
        kappa: Double = 2.0,
        betaMax: Double = DEFAULT_BETA_MAX,
        isCancelled: () -> Boolean = { false },
    ): FloatImage {
        require(lam > 0.0) { "lam must be > 0" }
        val w = img.width
        val h = img.height
        val n = w * h
        val fft = Fft2DF32(h, w)

        val n1Re = planes(n)
        val n1Im = planes(n)
        for (c in 0 until 3) {
            val src = img.plane(c)
            for (i in 0 until n) n1Re[c][i] = src[i]
        }
        fft.transformChannels(n1Re, n1Im, false)

        val otf = computeOtf(fft, h, w)
        val invOtf = FloatArray(n)

        var s = planes(n)
        for (c in 0 until 3) {
            val src = img.plane(c)
            for (i in 0 until n) s[c][i] = src[i]
        }
        val hDiff = planes(n)
        val vDiff = planes(n)
        var workRe = planes(n)
        val workIm = planes(n)

        var beta = 2.0 * lam
        while (beta < betaMax) {
            if (isCancelled()) throw L0CancelledException()
            val threshold = (lam / beta).toFloat()

            Parallel.forEachRange(n) { from, to ->
                for (i in from until to) invOtf[i] = (1.0 / (1.0 + beta * otf[i])).toFloat()
            }

            // Rows are split as one contiguous block per plane, so every leaf runs a straight
            // loop over whole rows instead of a lambda per row.
            Parallel.forEachRange(3 * h) { from, to ->
                var idx = from
                while (idx < to) {
                    val c = idx / h
                    val y = idx % h
                    forwardDiffRow(s[c], hDiff[c], y, w)
                    forwardDiffColumn(s[c], vDiff[c], y, w, h)
                    idx++
                }
            }
            Parallel.forEachRange(h) { from, to ->
                for (y in from until to) maskRow(hDiff, vDiff, y, w, threshold)
            }
            Parallel.forEachRange(3 * h) { from, to ->
                var idx = from
                while (idx < to) {
                    val c = idx / h
                    val y = idx % h
                    divergenceRow(hDiff[c], vDiff[c], workRe[c], workIm[c], y, w, h)
                    idx++
                }
            }

            fft.transformChannels(workRe, workIm, false)
            val b = beta.toFloat()
            Parallel.forEachRange(3 * n) { from, to ->
                var idx = from
                while (idx < to) {
                    val c = idx / n
                    val i = idx - c * n
                    val k = invOtf[i]
                    workRe[c][i] = (n1Re[c][i] + b * workRe[c][i]) * k
                    workIm[c][i] = (n1Im[c][i] + b * workIm[c][i]) * k
                    idx++
                }
            }
            fft.transformChannels(workRe, workIm, true)

            Parallel.forEachRange(3 * h) { from, to ->
                var idx = from
                while (idx < to) {
                    val c = idx / h
                    val y = idx % h
                    val base = y * w
                    val sc = s[c]
                    val wc = workRe[c]
                    for (x in 0 until w) sc[base + x] = wc[base + x].toFloat()
                    idx++
                }
            }
            beta *= kappa
        }

        val out = FloatImage(w, h)
        for (c in 0 until 3) {
            val dst = out.plane(c)
            val src = s[c]
            for (i in 0 until n) {
                val v = src[i]
                dst[i] = if (v < 0f) 0f else if (v > 1f) 1f else v
            }
        }
        return out
    }

    private fun planes(n: Int) = arrayOf(FloatArray(n), FloatArray(n), FloatArray(n))

    private fun computeOtf(fft: Fft2DF32, h: Int, w: Int): FloatArray {
        val n = h * w
        val fxRe = FloatArray(n); val fxIm = FloatArray(n)
        val fyRe = FloatArray(n); val fyIm = FloatArray(n)
        fxRe[0] = -1f; fxRe[w - 1] = 1f
        fyRe[0] = -1f; fyRe[n - w] = 1f
        fft.transformChannels(arrayOf(fxRe, fyRe), arrayOf(fxIm, fyIm), false)
        val otf = FloatArray(n)
        for (i in 0 until n) {
            val ax = fxRe[i]; val ay = fxIm[i]
            val bx = fyRe[i]; val by = fyIm[i]
            otf[i] = ax * ax + ay * ay + bx * bx + by * by
        }
        return otf
    }

    private fun forwardDiffRow(src: FloatArray, dst: FloatArray, y: Int, w: Int) {
        val base = y * w
        val last = base + w - 1
        for (x in base until last) dst[x] = src[x + 1] - src[x]
        dst[last] = src[base] - src[last]
    }

    private fun forwardDiffColumn(src: FloatArray, dst: FloatArray, y: Int, w: Int, hgt: Int) {
        val base = y * w
        val next = if (y == hgt - 1) 0 else base + w
        for (x in 0 until w) dst[base + x] = src[next + x] - src[base + x]
    }

    private fun maskRow(hD: Array<FloatArray>, vD: Array<FloatArray>, y: Int, w: Int, threshold: Float) {
        val base = y * w
        for (x in 0 until w) {
            val i = base + x
            val energy = hD[0][i] * hD[0][i] + vD[0][i] * vD[0][i] +
                hD[1][i] * hD[1][i] + vD[1][i] * vD[1][i] +
                hD[2][i] * hD[2][i] + vD[2][i] * vD[2][i]
            if (energy < threshold) {
                hD[0][i] = 0f; vD[0][i] = 0f
                hD[1][i] = 0f; vD[1][i] = 0f
                hD[2][i] = 0f; vD[2][i] = 0f
            }
        }
    }

    private fun divergenceRow(
        hD: FloatArray,
        vD: FloatArray,
        outRe: FloatArray,
        outIm: FloatArray,
        y: Int,
        w: Int,
        hgt: Int,
    ) {
        val base = y * w
        val prevRow = if (y == 0) (hgt - 1) * w else base - w
        for (x in 0 until w) {
            val i = base + x
            val hPrev = if (x == 0) hD[base + w - 1] else hD[i - 1]
            outRe[i] = (hPrev - hD[i]) + (vD[prevRow + x] - vD[i])
            outIm[i] = 0f
        }
    }
}
