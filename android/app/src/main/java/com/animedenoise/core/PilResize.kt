package com.animedenoise.core

import kotlin.math.ceil
import kotlin.math.sin

/**
 * Byte-exact port of Pillow's 8-bit resampler, i.e. of `Image.resize` for RGB/RGBA images.
 *
 * The reference tool does its geometry with two PIL calls
 *
 * ```python
 * small = src.resize((round(W * 0.6), round(H * 0.6)), Image.BOX)      # denoise.sr_at
 * back  = small.resize((W, H), Image.LANCZOS)
 * ```
 *
 * so reproducing them exactly - not "approximately", not "in float" - is what lets the
 * Android output match the Python output pixel for pixel. Everything here therefore works
 * on the 8-bit domain with Pillow's own fixed-point coefficients:
 *
 *  * the filter coefficients are built in [Double] and immediately normalised to sum to 1,
 *    exactly as `precompute_coeffs` does;
 *  * they are then quantised to 32-bit integers with `PRECISION_BITS = 22`, which is
 *    `normalize_coeffs_8bpc`;
 *  * the convolution accumulates in a 32-bit [Int] that starts at `1 shl 21` (the rounding
 *    bias) and the result is `acc shr 22` clamped to 0..255 - exactly `clip8()`.
 *
 * The horizontal pass runs first, then the vertical pass, matching `ImagingResampleInner`.
 * Rows are independent, so both passes are spread over the worker pool.
 *
 * The filters:
 *
 * | filter | kernel | support |
 * |---|---|---|
 * | BOX | `1` if `-0.5 < x <= 0.5`, else `0` | 0.5 |
 * | LANCZOS | `sinc(x) * sinc(x / 3)` for `-3 <= x < 3`, else `0` | 3.0 |
 *
 * `BOX` only averages when shrinking (`filterscale = inSize / outSize`, clamped up to 1);
 * `LANCZOS` works in both directions and is used here to enlarge 60 % back to 100 %.
 *
 * ### Alpha
 *
 * Pillow resizes RGBA by converting to premultiplied `RGBa` first, resizing, then
 * converting back (`Image.resize` lines 2407-2410). That round-trip is reproduced so the
 * result is byte-identical to PIL's, including the precision loss it costs at partial
 * alpha. Fully opaque images are unaffected: at `a = 255` the round-trip is exact.
 *
 * [resizeStraight] is the mode-`RGB` path, with no alpha handling at all.
 *
 * ### What is not ported
 *
 * `Image.resize` also has a `reducing_gap` fast path (default `None`, unused here) and a
 * special case for images taller than 100x their width (line 2432). Neither is reachable
 * from the denoise pipeline.
 */
object PilResize {

    /** The subset of Pillow's resampling filters this port needs. */
    enum class Filter(val support: Double) {
        /** Area average (Pillow's `Image.Resampling.BOX`). */
        BOX(0.5),

        /** Windowed sinc with `a = 3` (Pillow's `Image.Resampling.LANCZOS`). */
        LANCZOS(3.0);

        /** Applies the kernel. Pillow passes `x / filterscale`, so callers scale first. */
        fun kernel(x: Double): Double = when (this) {
            BOX -> if (x > -0.5 && x <= 0.5) 1.0 else 0.0
            LANCZOS -> if (x >= -3.0 && x < 3.0) sinc(x) * sinc(x / 3.0) else 0.0
        }
    }

    /** `sinc_filter` from Pillow's Resample.c. */
    private fun sinc(x: Double): Double {
        if (x == 0.0) return 1.0
        val v = x * Math.PI
        return sin(v) / v
    }

    /**
     * Resizes [src] to exactly `dstW x dstH` pixels with Pillow's RGBA path.
     *
     * Alpha is premultiplied before the resample and unpremultiplied after, as Pillow does
     * for mode `RGBA`.
     *
     * `param src    source image in ARGB_8888 byte order
     * `param dstW   target width, must be positive
     * `param dstH   target height, must be positive
     * `param filter [Filter.BOX] or [Filter.LANCZOS]
     * `return a new image; [src] is never modified
     */
    fun resize(src: RgbaImage, dstW: Int, dstH: Int, filter: Filter): RgbaImage {
        require(dstW > 0 && dstH > 0) { "target size must be positive, was " + dstW + "x" + dstH }
        if (src.width == dstW && src.height == dstH) {
            return RgbaImage(src.width, src.height, src.pixels.copyOf())
        }
        val resized = Resampler(src.width, src.height, dstW, dstH, filter).run(premultiply(src))
        return unpremultiply(resized)
    }

    /**
     * Resamples [src] with no alpha handling, i.e. Pillow's mode-`RGB` path.
     *
     * This is what the reference pipeline actually exercises, because `denoise.py` calls
     * `Image.convert("RGB")` before any geometry. Use it to compare against a PIL RGB
     * resize, or when the alpha channel must pass through untouched.
     */
    /**
     * Resamples only output rows [row0, row1) of a full resize.
     *
     * Built for the streaming pipeline: it needs one horizontal strip of the resized image and must
     * not allocate the whole thing. The horizontal pass runs over the source rows that feed those
     * output rows only, and the vertical pass is limited to the requested rows. Coeff tables and the
     * fixed-point arithmetic are the same ones [resize] uses, so a strip is bit-identical to the
     * corresponding rows of a whole-image resize.
     *
     * @param outWidth width of the destination image
     * @param outHeight height of the destination image (needed for the vertical coeff table)
     */
    fun resizeRowBand(
        src: RgbaImage,
        outWidth: Int,
        outHeight: Int,
        row0: Int,
        row1: Int,
        filter: Filter,
    ): RgbaImage {
        val inW = src.width
        val inH = src.height

        // Horizontal pass for the whole input height would defeat the purpose, so first work out
        // which input rows the requested output rows need.
        val vc = precomputeCoeffs(0.0, inH.toDouble(), outHeight, filter)
        val kv = normalizeCoeffs(vc)
        var srcLo = Int.MAX_VALUE
        var srcHi = 0
        for (yy in row0 until row1) {
            val ymin = vc.bounds[yy * 2]
            val ymax = vc.bounds[yy * 2 + 1]
            if (ymax > 0) {
                srcLo = minOf(srcLo, ymin)
                srcHi = maxOf(srcHi, ymin + ymax)
            }
        }
        if (srcLo >= srcHi) { srcLo = 0; srcHi = minOf(inH, 1) }

        // Horizontal pass over just those rows.
        val hc = precomputeCoeffs(0.0, inW.toDouble(), outWidth, filter)
        val kh = normalizeCoeffs(hc)
        val bandRows = srcHi - srcLo
        val horiz = ByteArray(outWidth * bandRows * 4)
        val stride = outWidth * 4
        val srcStride = inW * 4
        val srcPx = src.pixels
        val boundsH = hc.bounds
        val ksizeH = hc.ksize
        Parallel.forEach(bandRows) { y ->
            val inRow = (srcLo + y) * srcStride
            val outRow = y * stride
            for (xx in 0 until outWidth) {
                val xmin = boundsH[xx * 2]
                val xmax = boundsH[xx * 2 + 1]
                val kb = xx * ksizeH
                var s0 = 1 shl (PRECISION_BITS - 1)
                var s1 = s0; var s2 = s0; var s3 = s0
                for (x in 0 until xmax) {
                    val o = inRow + (x + xmin) * 4
                    val kvv = kh[kb + x]
                    s0 += (srcPx[o].toInt() and 0xFF) * kvv
                    s1 += (srcPx[o + 1].toInt() and 0xFF) * kvv
                    s2 += (srcPx[o + 2].toInt() and 0xFF) * kvv
                    s3 += (srcPx[o + 3].toInt() and 0xFF) * kvv
                }
                val d = outRow + xx * 4
                horiz[d] = clip8Value(s0)
                horiz[d + 1] = clip8Value(s1)
                horiz[d + 2] = clip8Value(s2)
                horiz[d + 3] = clip8Value(s3)
            }
        }

        // Vertical pass for the requested rows only, reading from the cropped band.
        val rows = row1 - row0
        val out = ByteArray(outWidth * rows * 4)
        val ksizeV = vc.ksize
        val boundsV = vc.bounds
        Parallel.forEach(rows) { r ->
            val yy = row0 + r
            val ymin = boundsV[yy * 2]
            val ymax = boundsV[yy * 2 + 1]
            val kb = yy * ksizeV
            val outRow = r * stride
            for (xx in 0 until outWidth) {
                var s0 = 1 shl (PRECISION_BITS - 1)
                var s1 = s0; var s2 = s0; var s3 = s0
                val col = xx * 4
                for (y in 0 until ymax) {
                    val o = (ymin + y - srcLo) * stride + col
                    val kvv = kv[kb + y]
                    s0 += (horiz[o].toInt() and 0xFF) * kvv
                    s1 += (horiz[o + 1].toInt() and 0xFF) * kvv
                    s2 += (horiz[o + 2].toInt() and 0xFF) * kvv
                    s3 += (horiz[o + 3].toInt() and 0xFF) * kvv
                }
                val d = outRow + col
                out[d] = clip8Value(s0)
                out[d + 1] = clip8Value(s1)
                out[d + 2] = clip8Value(s2)
                out[d + 3] = clip8Value(s3)
            }
        }
        return RgbaImage(outWidth, rows, out)
    }

    fun resizeStraight(src: RgbaImage, dstW: Int, dstH: Int, filter: Filter): RgbaImage {
        require(dstW > 0 && dstH > 0) { "target size must be positive, was " + dstW + "x" + dstH }
        if (src.width == dstW && src.height == dstH) {
            return RgbaImage(src.width, src.height, src.pixels.copyOf())
        }
        return Resampler(src.width, src.height, dstW, dstH, filter).run(src)
    }

    /**
     * The size Pillow is asked for when shrinking by [scale] - i.e. Python 3's `round()`.
     *
     * `round()` is a *banker's* rounding: halves go to the nearest even integer, unlike
     * `Math.round` or `(int)(x + 0.5)`. `round(5 * 0.5)` is 2, not 3, and
     * `round(1024 * 0.6)` is 614. The `denoise.py` call sites are `round(W * DOWN)` and
     * `round(H * DOWN)` in `sr_at`, with `DOWN = 0.6`.
     *
     * `param size  source dimension in pixels
     * `param scale multiplier, `0.6` for the default pipeline
     * `return the scaled dimension, never below 1
     */
    fun pythonRoundSize(size: Int, scale: Double): Int {
        val v = size * scale
        val floor = Math.floor(v)
        val diff = v - floor
        // Exactly representable halves take the even neighbour; everything else rounds like
        // normal. boxed(Math.floorMod) keeps the even test correct for negative floors.
        val rounded = when {
            diff > 0.5 -> floor + 1.0
            diff < 0.5 -> floor
            (floor.toLong() % 2L) == 0L -> floor
            else -> floor + 1.0
        }
        val result = rounded.toInt()
        return if (result < 1) 1 else result
    }

    // ------------------------------------------------------------------ alpha

    /** `rgbA2rgba` from Convert.c: premultiply RGB by A with Pillow's MULDIV255. */
    private fun premultiply(src: RgbaImage): RgbaImage {
        val n = src.width * src.height
        val pin = src.pixels
        val out = ByteArray(pin.size)
        for (i in 0 until n) {
            val o = i * 4
            val a = pin[o + 3].toInt() and 0xFF
            out[o] = mulDiv255(pin[o].toInt() and 0xFF, a).toByte()
            out[o + 1] = mulDiv255(pin[o + 1].toInt() and 0xFF, a).toByte()
            out[o + 2] = mulDiv255(pin[o + 2].toInt() and 0xFF, a).toByte()
            out[o + 3] = a.toByte()
        }
        return RgbaImage(src.width, src.height, out)
    }

    /** `rgba2rgbA` from Convert.c: undo the premultiplication, with Pillow's clipping. */
    private fun unpremultiply(src: RgbaImage): RgbaImage {
        val n = src.width * src.height
        val pin = src.pixels
        val out = ByteArray(pin.size)
        for (i in 0 until n) {
            val o = i * 4
            val a = pin[o + 3].toInt() and 0xFF
            if (a == 255 || a == 0) {
                out[o] = pin[o]
                out[o + 1] = pin[o + 1]
                out[o + 2] = pin[o + 2]
            } else {
                out[o] = clip8((255 * (pin[o].toInt() and 0xFF)) / a)
                out[o + 1] = clip8((255 * (pin[o + 1].toInt() and 0xFF)) / a)
                out[o + 2] = clip8((255 * (pin[o + 2].toInt() and 0xFF)) / a)
            }
            out[o + 3] = a.toByte()
        }
        return RgbaImage(src.width, src.height, out)
    }

    /** `MULDIV255(a, b, tmp)`: `(a * b + 127) / 255` without a division. */
    private fun mulDiv255(a: Int, b: Int): Int {
        val tmp = a * b + 128
        return ((tmp shr 8) + tmp) shr 8
    }

    /** `CLIP8(v)` from ImagingUtils.h. */
    private fun clip8(v: Int): Byte = (if (v <= 0) 0 else if (v < 256) v else 255).toByte()

    // -------------------------------------------------------------- resampler

    /** The per-output-pixel coefficient windows Pillow builds in `precompute_coeffs`. */
    private class Coefficients(
        val outSize: Int,
        val ksize: Int,
        val bounds: IntArray,
        val kk: DoubleArray
    )

    /**
     * Builds Pillow's coefficient table for one axis.
     *
     * `in0`/`in1` are the source box (Pillow's `box[0]`/`box[2]`, and the vertical pair).
     * The pipeline always uses the whole image, so they are 0 and `inSize`, but they are
     * kept because they change `filterscale` and therefore the window.
     */
    private fun precomputeCoeffs(in0: Double, in1: Double, outSize: Int, filter: Filter): Coefficients {
        val inSize = in1.toInt()
        var filterscale = (in1 - in0) / outSize
        if (filterscale < 1.0) filterscale = 1.0
        val support = filter.support * filterscale
        val ksize = ceil(support).toInt() * 2 + 1
        val bounds = IntArray(outSize * 2)
        val kk = DoubleArray(outSize * ksize)
        val invFilterscale = 1.0 / filterscale
        val scale = (in1 - in0) / outSize
        for (xx in 0 until outSize) {
            val center = in0 + (xx + 0.5) * scale
            var xmin = (center - support + 0.5).toInt()
            if (xmin < 0) xmin = 0
            var xmax = (center + support + 0.5).toInt()
            if (xmax > inSize) xmax = inSize
            xmax -= xmin
            val base = xx * ksize
            var ww = 0.0
            for (x in 0 until xmax) {
                val w = filter.kernel((x + xmin - center + 0.5) * invFilterscale)
                kk[base + x] = w
                ww += w
            }
            if (ww != 0.0) {
                for (x in 0 until xmax) kk[base + x] /= ww
            }
            for (x in xmax until ksize) kk[base + x] = 0.0
            bounds[xx * 2] = xmin
            bounds[xx * 2 + 1] = xmax
        }
        return Coefficients(outSize, ksize, bounds, kk)
    }

    /** `normalize_coeffs_8bpc`: doubles to 32-bit fixed point with `PRECISION_BITS` fraction. */
    private fun normalizeCoeffs(c: Coefficients): IntArray {
        val out = IntArray(c.outSize * c.ksize)
        val k = c.kk
        val scale = (1 shl PRECISION_BITS).toDouble()
        for (i in out.indices) {
            val v = k[i] * scale
            // C's (int) cast truncates toward zero; Pillow adds the +/-0.5 bias first.
            out[i] = if (k[i] < 0) (-0.5 + v).toInt() else (0.5 + v).toInt()
        }
        return out
    }

    /** One resize job: horizontal pass then vertical pass, like `ImagingResampleInner`. */
    private class Resampler(
        private val srcW: Int,
        private val srcH: Int,
        private val dstW: Int,
        private val dstH: Int,
        private val filter: Filter
    ) {

        fun run(src: RgbaImage): RgbaImage {
            var input = src.pixels
            var inW = srcW
            var inH = srcH

            if (dstW != srcW) {
                val coeffs = precomputeCoeffs(0.0, srcW.toDouble(), dstW, filter)
                val k = normalizeCoeffs(coeffs)
                val outPixels = ByteArray(dstW * inH * 4)
                resampleHorizontal(input, inW, inH, outPixels, coeffs, k)
                input = outPixels
                inW = dstW
            }
            if (dstH != srcH) {
                val coeffs = precomputeCoeffs(0.0, srcH.toDouble(), dstH, filter)
                val k = normalizeCoeffs(coeffs)
                val outPixels = ByteArray(inW * dstH * 4)
                resampleVertical(input, inW, inH, outPixels, coeffs, k)
                input = outPixels
                inH = dstH
            }
            return RgbaImage(inW, inH, input)
        }

        /** `_ImagingResampleHorizontal_8bpc`, four bands. */
        private fun resampleHorizontal(
            input: ByteArray,
            inW: Int,
            inH: Int,
            out: ByteArray,
            c: Coefficients,
            k: IntArray
        ) {
            val bounds = c.bounds
            val ksize = c.ksize
            val outW = c.outSize
            Parallel.forEach(inH) { y ->
                val inRow = y * inW * 4
                val outRow = y * outW * 4
                for (xx in 0 until outW) {
                    val xmin = bounds[xx * 2]
                    val xmax = bounds[xx * 2 + 1]
                    val kb = xx * ksize
                    var s0 = 1 shl (PRECISION_BITS - 1)
                    var s1 = s0
                    var s2 = s0
                    var s3 = s0
                    for (x in 0 until xmax) {
                        val o = inRow + (x + xmin) * 4
                        val kv = k[kb + x]
                        s0 += (input[o].toInt() and 0xFF) * kv
                        s1 += (input[o + 1].toInt() and 0xFF) * kv
                        s2 += (input[o + 2].toInt() and 0xFF) * kv
                        s3 += (input[o + 3].toInt() and 0xFF) * kv
                    }
                    val d = outRow + xx * 4
                    out[d] = clip8Value(s0)
                    out[d + 1] = clip8Value(s1)
                    out[d + 2] = clip8Value(s2)
                    out[d + 3] = clip8Value(s3)
                }
            }
        }

        /** `_ImagingResampleVertical_8bpc`, four bands. */
        private fun resampleVertical(
            input: ByteArray,
            inW: Int,
            inH: Int,
            out: ByteArray,
            c: Coefficients,
            k: IntArray
        ) {
            val bounds = c.bounds
            val ksize = c.ksize
            val outH = c.outSize
            val rowStride = inW * 4
            Parallel.forEach(outH) { yy ->
                val ymin = bounds[yy * 2]
                val ymax = bounds[yy * 2 + 1]
                val kb = yy * ksize
                val outRow = yy * rowStride
                for (xx in 0 until inW) {
                    var s0 = 1 shl (PRECISION_BITS - 1)
                    var s1 = s0
                    var s2 = s0
                    var s3 = s0
                    val col = xx * 4
                    for (y in 0 until ymax) {
                        val o = (y + ymin) * rowStride + col
                        val kv = k[kb + y]
                        s0 += (input[o].toInt() and 0xFF) * kv
                        s1 += (input[o + 1].toInt() and 0xFF) * kv
                        s2 += (input[o + 2].toInt() and 0xFF) * kv
                        s3 += (input[o + 3].toInt() and 0xFF) * kv
                    }
                    val d = outRow + col
                    out[d] = clip8Value(s0)
                    out[d + 1] = clip8Value(s1)
                    out[d + 2] = clip8Value(s2)
                    out[d + 3] = clip8Value(s3)
                }
            }
        }
    }

    /** `clip8(int)`: `acc shr 22` saturated to 0..255. */
    private fun clip8Value(acc: Int): Byte {
        val v = acc shr PRECISION_BITS
        return (if (v < 0) 0 else if (v > 255) 255 else v).toByte()
    }

    /** `#define PRECISION_BITS (32 - 8 - 2)` from Resample.c. */
    private const val PRECISION_BITS = 22
}
