package com.animedenoise.core

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * A complex FFT of one fixed length with every twiddle factor precomputed.
 *
 * Supported lengths - picked automatically, the caller never chooses an algorithm:
 *
 *  * powers of two: iterative radix-2 Cooley-Tukey with a bit-reversal permutation;
 *  * @p * m@ where @p@ is 3, 5 or 7 and @m@ is a power of two (1536 = 3 * 512, 1920 = 5 * 384):
 *    one Cooley-Tukey step of radix @p@ followed by @p@ smaller transforms, built recursively;
 *  * anything else - a prime width such as 941, or a size with a large prime factor:
 *    Bluestein's chirp-z algorithm on top of a power-of-two transform.
 *
 * Real images have odd dimensions all the time (1672x941 is one of the sizes the reference
 * tool documents), so all three paths matter.
 *
 * The plan is immutable after construction and one instance may be shared by any number of
 * threads, as long as each call gets its own buffers. Scratch space is kept per thread.
 *
 * All arithmetic is [Double], matching the float64 code path of the Python reference.
 *
 * @param n transform length in samples, must be positive
 */
class Fft1D(val n: Int) {

    private var kind = KIND_RADIX2

    // Radix-2 plan.
    private var rev = IntArray(0)
    private var twRe = DoubleArray(0)
    private var twIm = DoubleArray(0)
    private var twReInv = DoubleArray(0)
    private var twImInv = DoubleArray(0)

    // Mixed-radix plan: n = radix * subLength, with a sub-plan of length subLength.
    private var radix = 0
    private var subLength = 0
    private var sub: Fft1D? = null
    private var phaseRe = DoubleArray(0)
    private var phaseIm = DoubleArray(0)
    private var smallRe = DoubleArray(0)
    private var smallIm = DoubleArray(0)

    // Bluestein plan.
    private var blueLength = 0
    private var blueInner: Fft1D? = null
    private var chirpRe = DoubleArray(0)
    private var chirpIm = DoubleArray(0)
    private var kernelRe = DoubleArray(0)
    private var kernelIm = DoubleArray(0)

    private val scratch: ThreadLocal<DoubleArray2>

    init {
        require(n >= 1) { "FFT length must be positive, was " + n }
        if (isPowerOfTwo(n)) {
            kind = KIND_RADIX2
            buildRadix2()
        } else {
            val p = chooseRadix(n)
            if (p != 0) {
                kind = KIND_MIXED
                radix = p
                subLength = n / p
                sub = Fft1D(subLength)
                buildMixed()
            } else {
                kind = KIND_BLUESTEIN
                buildBluestein()
            }
        }
        val scratchSize = if (kind == KIND_BLUESTEIN) blueLength else n
        scratch = ThreadLocal.withInitial { DoubleArray2(DoubleArray(scratchSize), DoubleArray(scratchSize)) }
    }

    /**
     * Transforms the @n@ samples starting at [offset] in place.
     *
     * When [inverse] is true the result is scaled by @1 / n@, the same convention as
     * @np.fft.ifft@.
     */
    fun transform(re: DoubleArray, im: DoubleArray, offset: Int = 0, inverse: Boolean = false) {
        require(offset >= 0 && offset + n <= re.size && offset + n <= im.size) {
            "buffer too small for an FFT of length " + n + " at offset " + offset
        }
        transformUnscaled(re, im, offset, inverse)
        if (inverse) {
            val s = 1.0 / n
            for (i in offset until offset + n) {
                re[i] *= s
                im[i] *= s
            }
        }
    }

    /** Transform without the @1 / n@ scaling; the mixed-radix step uses this internally. */
    private fun transformUnscaled(re: DoubleArray, im: DoubleArray, offset: Int, inverse: Boolean) {
        when (kind) {
            KIND_RADIX2 -> radix2(re, im, offset, inverse)
            KIND_MIXED -> mixedRadix(re, im, offset, inverse)
            else -> {
                if (inverse) {
                    // No inverse Bluestein plan is needed: IDFT(x) = conj(DFT(conj(x))) / n,
                    // and the 1 / n scaling is applied by [transform] itself.
                    conjugate(im, offset)
                    bluesteinForward(re, im, offset)
                    conjugate(im, offset)
                } else {
                    bluesteinForward(re, im, offset)
                }
            }
        }
    }

    // ---------------------------------------------------------------- radix-2

    private fun buildRadix2() {
        val bits = Integer.numberOfTrailingZeros(n)
        rev = IntArray(n)
        for (i in 0 until n) {
            rev[i] = Integer.reverse(i) ushr (32 - bits)
        }
        val half = n / 2
        twRe = DoubleArray(half)
        twIm = DoubleArray(half)
        twReInv = DoubleArray(half)
        twImInv = DoubleArray(half)
        for (k in 0 until half) {
            val ang = -2.0 * PI * k / n
            twRe[k] = cos(ang)
            twIm[k] = sin(ang)
            twReInv[k] = twRe[k]
            twImInv[k] = -twIm[k]
        }
    }

    private fun radix2(re: DoubleArray, im: DoubleArray, offset: Int, inverse: Boolean) {
        val rv = rev
        for (i in 0 until n) {
            val j = rv[i]
            if (j > i) {
                val a = offset + i
                val b = offset + j
                val tr = re[a]
                re[a] = re[b]
                re[b] = tr
                val ti = im[a]
                im[a] = im[b]
                im[b] = ti
            }
        }
        // Both twiddle tables are precomputed, so the butterfly has no sign multiply.
        val wr3 = if (inverse) twReInv else twRe
        val wi3 = if (inverse) twImInv else twIm
        var len = 2
        while (len <= n) {
            val halfLen = len shr 1
            val step = n / len
            var i = 0
            while (i < n) {
                var k = 0
                var j = 0
                while (j < halfLen) {
                    val wr = wr3[k]
                    val wi = wi3[k]
                    val a = offset + i + j
                    val b = a + halfLen
                    val br = re[b]
                    val bi = im[b]
                    val xr = br * wr - bi * wi
                    val xi = br * wi + bi * wr
                    re[b] = re[a] - xr
                    im[b] = im[a] - xi
                    re[a] += xr
                    im[a] += xi
                    j++
                    k += step
                }
                i += len
            }
            len = len shl 1
        }
    }

    // ------------------------------------------------------------ mixed radix

    /** Picks a small odd radix for [n], preferring one that leaves a power of two. */
    private fun chooseRadix(n: Int): Int {
        for (p in intArrayOf(3, 5, 7)) {
            if (n % p == 0 && isPowerOfTwo(n / p)) return p
        }
        for (p in intArrayOf(3, 5, 7)) {
            if (n % p == 0) return p
        }
        return 0
    }

    private fun buildMixed() {
        val p = radix
        val m = subLength
        phaseRe = DoubleArray(n)
        phaseIm = DoubleArray(n)
        for (j1 in 0 until m) {
            for (r in 0 until p) {
                val e = (j1.toLong() * r) % n
                val ang = -2.0 * PI * e / n
                phaseRe[j1 * p + r] = cos(ang)
                phaseIm[j1 * p + r] = sin(ang)
            }
        }
        smallRe = DoubleArray(p * p)
        smallIm = DoubleArray(p * p)
        for (j2 in 0 until p) {
            for (r in 0 until p) {
                val e = (j2 * r) % p
                val ang = -2.0 * PI * e / p
                smallRe[j2 * p + r] = cos(ang)
                smallIm[j2 * p + r] = sin(ang)
            }
        }
    }

    /**
     * Cooley-Tukey with @n = radix * subLength@.
     *
     * Stream 1: for every sub-sample @j1@ a @radix@-point DFT along the stride-@m@
     * direction, multiplied by @exp(-2 pi i j1 r / n)@. Stream 2: @radix@ independent
     * length-@subLength@ transforms, scattered with stride @radix@.
     */
    private fun mixedRadix(re: DoubleArray, im: DoubleArray, offset: Int, inverse: Boolean) {
        val p = radix
        val m = subLength
        val inner = sub!!
        val sc = scratch.get()
        val br = sc.re
        val bi = sc.im
        // smallIm / phaseIm hold sin(-angle), so forward uses them unmodified.
        val sign = if (inverse) -1.0 else 1.0

        for (j1 in 0 until m) {
            val base = offset + j1
            for (r in 0 until p) {
                var ar = 0.0
                var ai = 0.0
                for (j2 in 0 until p) {
                    val idx = base + m * j2
                    val xr = re[idx]
                    val xi = im[idx]
                    val wr = smallRe[j2 * p + r]
                    val wi = sign * smallIm[j2 * p + r]
                    ar += xr * wr - xi * wi
                    ai += xr * wi + xi * wr
                }
                val twr = phaseRe[j1 * p + r]
                val twi = sign * phaseIm[j1 * p + r]
                br[r * m + j1] = ar * twr - ai * twi
                bi[r * m + j1] = ar * twi + ai * twr
            }
        }
        for (r in 0 until p) {
            inner.transformUnscaled(br, bi, r * m, inverse)
            val src = r * m
            for (t in 0 until m) {
                val o = offset + r + p * t
                re[o] = br[src + t]
                im[o] = bi[src + t]
            }
        }
    }

    // ------------------------------------------------------------- Bluestein

    private fun buildBluestein() {
        blueLength = nextPowerOfTwo(2 * n - 1)
        blueInner = Fft1D(blueLength)
        // The transform puts a small negative phase on every iteration, so the forward
        // kernel is exp(-i*pi*j^2/n) and the inverse (which is just the conjugate of the
        // forward) conjugates it below. Reducing j^2 modulo 2n keeps the phase exact.
        chirpRe = DoubleArray(n)
        chirpIm = DoubleArray(n)
        for (j in 0 until n) {
            val e = (j.toLong() * j) % (2L * n)
            val ang = -2.0 * PI * e / (2.0 * n)
            chirpRe[j] = cos(ang)
            chirpIm[j] = sin(ang)
        }
        // Circular kernel: the linear convolution b[j] = conj(c[j]) is folded into the
        // zero-padded length with b[0] = conj(c[0]) and b[blueLength - j] = b[j].
        kernelRe = DoubleArray(blueLength)
        kernelIm = DoubleArray(blueLength)
        kernelRe[0] = chirpRe[0]
        kernelIm[0] = -chirpIm[0]
        for (j in 1 until n) {
            val cr = chirpRe[j]
            val ci = -chirpIm[j]
            kernelRe[j] = cr
            kernelIm[j] = ci
            kernelRe[blueLength - j] = cr
            kernelIm[blueLength - j] = ci
        }
        blueInner?.transform(kernelRe, kernelIm, 0, false)
    }

    private fun bluesteinForward(re: DoubleArray, im: DoubleArray, offset: Int) {
        val inner = blueInner!!
        val sc = scratch.get()
        val ar = sc.re
        val ai = sc.im
        java.util.Arrays.fill(ar, 0.0)
        java.util.Arrays.fill(ai, 0.0)
        for (j in 0 until n) {
            val cr = chirpRe[j]
            val ci = chirpIm[j]
            val xr = re[offset + j]
            val xi = im[offset + j]
            ar[j] = xr * cr - xi * ci
            ai[j] = xr * ci + xi * cr
        }
        inner.transform(ar, ai, 0, false)
        for (j in 0 until blueLength) {
            val br = kernelRe[j]
            val bi = kernelIm[j]
            val xr = ar[j]
            val xi = ai[j]
            ar[j] = xr * br - xi * bi
            ai[j] = xr * bi + xi * br
        }
        inner.transform(ar, ai, 0, true)
        // X[k] = c[k] * (a * b)[k], with c[k] = exp(-i*pi*k^2/n) as built above.
        for (k in 0 until n) {
            val cr = chirpRe[k]
            val ci = chirpIm[k]
            val xr = ar[k]
            val xi = ai[k]
            re[offset + k] = xr * cr - xi * ci
            im[offset + k] = xr * ci + xi * cr
        }
    }

    private fun conjugate(im: DoubleArray, offset: Int) {
        for (i in offset until offset + n) im[i] = -im[i]
    }

    private companion object {
        const val KIND_RADIX2 = 0
        const val KIND_MIXED = 1
        const val KIND_BLUESTEIN = 2

        fun isPowerOfTwo(v: Int): Boolean = v > 0 && (v and (v - 1)) == 0

        fun nextPowerOfTwo(v: Int): Int {
            var p = 1
            while (p < v) p = p shl 1
            return p
        }
    }
}

/**
 * A two-dimensional complex FFT of a fixed @height x width@ grid.
 *
 * Rows are transformed as contiguous runs, then columns are gathered into a per-thread
 * scratch line and transformed - the same decomposition as numpy's @fft2@ along axes
 * (0, 1). Every plane, every row and every column is an independent work item, so all of
 * them are spread over the worker pool: for a 1024x1536 image that is 1536 items in the row
 * pass and 1024 in the column pass, times as many channels as are batched.
 *
 * The inverse is scaled by @1 / (height * width)@, matching numpy's @ifft2@.
 *
 * Layout is row-major: sample @(x, y)@ lives at @y * width + x@ in both arrays.
 */
class Fft2D(val height: Int, val width: Int) {

    private val rowFft = Fft1D(width)
    private val columnFft = Fft1D(height)
    private val columnScratch = ThreadLocal.withInitial {
        DoubleArray2(DoubleArray(height), DoubleArray(height))
    }

    /** In-place 2D forward transform of one plane. */
    fun forward(re: DoubleArray, im: DoubleArray) {
        transformChannels(arrayOf(re), arrayOf(im), false)
    }

    /** In-place 2D inverse transform of one plane, scaled by @1 / (height * width)@. */
    fun inverse(re: DoubleArray, im: DoubleArray) {
        transformChannels(arrayOf(re), arrayOf(im), true)
    }

    /**
     * Transforms several equally sized planes at once, one plane per array pair.
     *
     * Batching the three colour planes is what keeps the pool busy: the row pass then has
     * @channels * height@ items and the column pass @channels * width@.
     */
    /** Double-precision entry point; [L0SmootherF32] stores float planes and promotes per call. */
    fun transformChannelsD(re: Array<DoubleArray>, im: Array<DoubleArray>, inverse: Boolean) {
        transformChannels(re, im, inverse)
    }

    /**
     * Float-plane entry point for the fast L0 path: promotes to double, runs the verified
     * transform, and stores the result back as float. Storage is what costs memory at 8 MP;
     * the butterflies stay double so no new numerics are introduced.
     */
    fun transformChannelsF(re: Array<FloatArray>, im: Array<FloatArray>, inverse: Boolean) {
        val n = height * width
        val dre = Array(re.size) { DoubleArray(n) }
        val dim = Array(im.size) { DoubleArray(n) }
        for (c in re.indices) {
            val fr = re[c]
            val fi = im[c]
            val dr = dre[c]
            val di = dim[c]
            for (i in 0 until n) {
                dr[i] = fr[i].toDouble()
                di[i] = fi[i].toDouble()
            }
        }
        transformChannels(dre, dim, inverse)
        for (c in re.indices) {
            val fr = re[c]
            val fi = im[c]
            val dr = dre[c]
            val di = dim[c]
            for (i in 0 until n) {
                fr[i] = dr[i].toFloat()
                fi[i] = di[i].toFloat()
            }
        }
    }

    fun transformChannels(re: Array<DoubleArray>, im: Array<DoubleArray>, inverse: Boolean) {
        val channels = re.size
        require(im.size == channels) { "re/im channel count mismatch" }
        if (channels == 0) return
        val n = height * width
        for (c in 0 until channels) {
            require(re[c].size >= n && im[c].size >= n) {
                "plane " + c + " is smaller than " + width + "x" + height
            }
        }

        Parallel.forEach(channels * height) { idx ->
            val c = idx / height
            val y = idx % height
            rowFft.transform(re[c], im[c], y * width, inverse)
        }
        // Columns are strided in the row-major layout, so each column is gathered into a
        // per-thread line, transformed, and scattered back. One item per column keeps the
        // pool busy (3 * width items) and the gather is O(height) sequential strides.
        Parallel.forEach(channels * width) { idx ->
            val c = idx / width
            val x = idx % width
            val sc = columnScratch.get()
            val rc = re[c]
            val ic = im[c]
            val sr = sc.re
            val si = sc.im
            for (y in 0 until height) {
                val o = y * width + x
                sr[y] = rc[o]
                si[y] = ic[o]
            }
            columnFft.transform(sr, si, 0, inverse)
            for (y in 0 until height) {
                val o = y * width + x
                rc[o] = sr[y]
                ic[o] = si[y]
            }
        }
    }
}

/** A complex plane as two parallel double arrays; the plans pass these around. */
internal class DoubleArray2(val re: DoubleArray, val im: DoubleArray)

