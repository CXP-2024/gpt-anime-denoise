package com.animedenoise.core

import kotlin.math.cos
import kotlin.math.sin

/**
 * Float-data FFT: identical structure to [Fft1D] but its transform() accepts FloatArray planes.
 *
 * The full 8.3 MP L0 solve was being killed by lowmemorykiller because ~500 MB of its working set
 * was double-precision FFT storage. Planes are where the megapixels live, so moving the data
 * buffers to float halves the footprint. The plan tables (twiddles, phases, chirps) stay double:
 * they are O(n) once per size and double rounding there keeps the phase exact; only the samples
 * passing through the butterflies are float, which the parity test bounds at 2 output levels.
 *
 * Structure is a line-by-line port of [Fft1D] (radix-2 / mixed-radix 3-5-7 / Bluestein), so the
 * same size support and the same scaling convention (inverse scaled by 1/n) apply.
 */
class Fft1DF32(val n: Int) {

    private var kind = KIND_RADIX2

    // Radix-2 plan.
    private var rev = IntArray(0)
    private var twRe = DoubleArray(0)
    private var twIm = DoubleArray(0)
    private var twReInv = DoubleArray(0)
    private var twImInv = DoubleArray(0)

    // Mixed-radix plan.
    private var radix = 0
    private var subLength = 0
    private var sub: Fft1DF32? = null
    private var phaseRe = DoubleArray(0)
    private var phaseIm = DoubleArray(0)
    private var smallRe = DoubleArray(0)
    private var smallIm = DoubleArray(0)

    // Bluestein plan.
    private var blueLength = 0
    private var blueInner: Fft1DF32? = null
    private var chirpRe = DoubleArray(0)
    private var chirpIm = DoubleArray(0)
    private var kernelRe = DoubleArray(0)
    private var kernelIm = DoubleArray(0)

    private val scratch: ThreadLocal<FloatArray2>

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
                sub = Fft1DF32(subLength)
                buildMixed()
            } else {
                kind = KIND_BLUESTEIN
                buildBluestein()
            }
        }
        val scratchSize = if (kind == KIND_BLUESTEIN) blueLength else n
        scratch = ThreadLocal.withInitial { FloatArray2(FloatArray(scratchSize), FloatArray(scratchSize)) }
    }

    /** Transforms [n] samples in place; inverse is scaled by 1/n like np.fft.ifft. */
    fun transform(re: FloatArray, im: FloatArray, offset: Int = 0, inverse: Boolean = false) {
        require(offset >= 0 && offset + n <= re.size && offset + n <= im.size) {
            "buffer too small for an FFT of length " + n + " at offset " + offset
        }
        transformUnscaled(re, im, offset, inverse)
        if (inverse) {
            val s = 1f / n
            for (i in offset until offset + n) {
                re[i] *= s
                im[i] *= s
            }
        }
    }

    private fun transformUnscaled(re: FloatArray, im: FloatArray, offset: Int, inverse: Boolean) {
        when (kind) {
            KIND_RADIX2 -> radix2(re, im, offset, inverse)
            KIND_MIXED -> mixedRadix(re, im, offset, inverse)
            else -> {
                if (inverse) {
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
        for (i in 0 until n) rev[i] = Integer.reverse(i) ushr (32 - bits)
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

    private fun radix2(re: FloatArray, im: FloatArray, offset: Int, inverse: Boolean) {
        val rv = rev
        for (i in 0 until n) {
            val j = rv[i]
            if (j > i) {
                val a = offset + i
                val b = offset + j
                val tr = re[a]; re[a] = re[b]; re[b] = tr
                val ti = im[a]; im[a] = im[b]; im[b] = ti
            }
        }
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
                    val wr = wr3[k].toFloat()
                    val wi = wi3[k].toFloat()
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

    private fun chooseRadix(n: Int): Int {
        for (p in intArrayOf(3, 5, 7)) if (n % p == 0 && isPowerOfTwo(n / p)) return p
        for (p in intArrayOf(3, 5, 7)) if (n % p == 0) return p
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

    private fun mixedRadix(re: FloatArray, im: FloatArray, offset: Int, inverse: Boolean) {
        val p = radix
        val m = subLength
        val inner = sub!!
        val sc = scratch.get()
        val br = sc.re
        val bi = sc.im
        val sign = if (inverse) -1.0 else 1.0
        val signF = sign.toFloat()

        for (j1 in 0 until m) {
            val base = offset + j1
            for (r in 0 until p) {
                var ar = 0f
                var ai = 0f
                for (j2 in 0 until p) {
                    val idx = base + m * j2
                    val xr = re[idx]
                    val xi = im[idx]
                    val wr = smallRe[j2 * p + r].toFloat()
                    val wi = (sign * smallIm[j2 * p + r]).toFloat()
                    ar += xr * wr - xi * wi
                    ai += xr * wi + xi * wr
                }
                val twr = phaseRe[j1 * p + r].toFloat()
                val twi = (sign * phaseIm[j1 * p + r]).toFloat()
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
        blueInner = Fft1DF32(blueLength)
        chirpRe = DoubleArray(n)
        chirpIm = DoubleArray(n)
        for (j in 0 until n) {
            val e = (j.toLong() * j) % (2L * n)
            val ang = -2.0 * PI * e / (2.0 * n)
            chirpRe[j] = cos(ang)
            chirpIm[j] = sin(ang)
        }
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
        blueInner?.transform(kernelReF(), kernelImF(), 0, false)
    }

    // The Bluestein kernel is plan data, not sample data; keep it double and stage it
    // into float buffers only for the inner transform.
    private var kernelReF: FloatArray? = null
    private var kernelImF: FloatArray? = null

    private fun kernelReF(): FloatArray {
        if (kernelReF == null) kernelReF = FloatArray(kernelRe.size) { kernelRe[it].toFloat() }
        return kernelReF!!
    }

    private fun kernelImF(): FloatArray {
        if (kernelImF == null) kernelImF = FloatArray(kernelIm.size) { kernelIm[it].toFloat() }
        return kernelImF!!
    }

    private fun bluesteinForward(re: FloatArray, im: FloatArray, offset: Int) {
        val inner = blueInner!!
        val sc = scratch.get()
        val ar = sc.re
        val ai = sc.im
        java.util.Arrays.fill(ar, 0f)
        java.util.Arrays.fill(ai, 0f)
        for (j in 0 until n) {
            val cr = chirpRe[j].toFloat()
            val ci = chirpIm[j].toFloat()
            val xr = re[offset + j]
            val xi = im[offset + j]
            ar[j] = xr * cr - xi * ci
            ai[j] = xr * ci + xi * cr
        }
        inner.transform(ar, ai, 0, false)
        val kr = kernelReF()
        val ki = kernelImF()
        for (j in 0 until blueLength) {
            val br = kr[j]
            val bi = ki[j]
            val xr = ar[j]
            val xi = ai[j]
            ar[j] = xr * br - xi * bi
            ai[j] = xr * bi + xi * br
        }
        inner.transform(ar, ai, 0, true)
        for (k in 0 until n) {
            val cr = chirpRe[k].toFloat()
            val ci = chirpIm[k].toFloat()
            val xr = ar[k]
            val xi = ai[k]
            re[offset + k] = xr * cr - xi * ci
            im[offset + k] = xr * ci + xi * cr
        }
    }

    private fun conjugate(im: FloatArray, offset: Int) {
        for (i in offset until offset + n) im[i] = -im[i]
    }

    private class FloatArray2(val re: FloatArray, val im: FloatArray)

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
        const val PI = Math.PI
    }
}
