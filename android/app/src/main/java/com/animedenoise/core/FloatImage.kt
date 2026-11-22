package com.animedenoise.core

/**
 * A planar RGB image in floating point, with the three colour planes stored separately.
 *
 * Storage is [Float] (32-bit) because the pipeline works on 8-bit source data and the
 * planes are only 1 MB per megapixel per channel, but **all arithmetic in the processing
 * nodes is carried out in [Double]** so that the result matches the Python/torch float64
 * reference path described in `denoise.py`.
 *
 * Pixel `(x, y)` of the red plane lives at index `y * width + x` (row-major), and the
 * same indexing is used for [g] and [b].
 *
 * Values are nominally in `[0, 1]` (a byte value of 255 maps to 1.0) but nothing enforces
 * that: nodes may produce out-of-range values and only the final quantisation clamps.
 *
 * This class holds no Android types, so it is usable from plain JVM unit tests.
 *
 * @property width image width in pixels, strictly positive
 * @property height image height in pixels, strictly positive
 */
class FloatImage(val width: Int, val height: Int) {

    /** Red plane, `width * height` samples, row-major. */
    val r: FloatArray

    /** Green plane, `width * height` samples, row-major. */
    val g: FloatArray

    /** Blue plane, `width * height` samples, row-major. */
    val b: FloatArray

    init {
        require(width > 0 && height > 0) { "empty image: " + width + "x" + height }
        val n = width * height
        r = FloatArray(n)
        g = FloatArray(n)
        b = FloatArray(n)
    }

    /** Number of samples per plane. */
    val sampleCount: Int get() = width * height

    /**
     * Returns the plane for [channel]: 0 = red, 1 = green, 2 = blue.
     *
     * The returned array is the live backing storage, not a copy.
     */
    fun plane(channel: Int): FloatArray = when (channel) {
        0 -> r
        1 -> g
        2 -> b
        else -> throw IllegalArgumentException("channel must be 0..2, was " + channel)
    }

    /** Deep copy, including all three planes. */
    fun copy(): FloatImage {
        val out = FloatImage(width, height)
        System.arraycopy(r, 0, out.r, 0, r.size)
        System.arraycopy(g, 0, out.g, 0, g.size)
        System.arraycopy(b, 0, out.b, 0, b.size)
        return out
    }

    override fun toString(): String = "FloatImage(" + width + "x" + height + ")"

    companion object {

        /**
         * Converts 8-bit RGBA into a [FloatImage] exactly like `denoise.to_arr`:
         *
         * ```python
         * np.asarray(im.convert("RGB"), np.float64) / 255
         * ```
         *
         * The alpha byte is read and **ignored**; RGB bytes are taken exactly as they are
         * (no premultiplication, no gamma handling). Each byte `v` becomes `v / 255.0`,
         * computed in double precision and then narrowed to the [Float] storage.
         */
        /** Builds a FloatImage from existing float planes (adopted, not copied). */
        fun fromPlanesF(width: Int, height: Int, r: FloatArray, g: FloatArray, b: FloatArray): FloatImage {
            require(r.size == width * height && g.size == width * height && b.size == width * height)
            val out = FloatImage(width, height)
            System.arraycopy(r, 0, out.r, 0, r.size)
            System.arraycopy(g, 0, out.g, 0, g.size)
            System.arraycopy(b, 0, out.b, 0, b.size)
            return out
        }

        fun fromRgba(src: RgbaImage): FloatImage {
            val w = src.width
            val h = src.height
            val n = w * h
            val out = FloatImage(w, h)
            val px = src.pixels
            for (i in 0 until n) {
                val o = i * 4
                out.r[i] = ((px[o].toInt() and 0xFF) / 255.0).toFloat()
                out.g[i] = ((px[o + 1].toInt() and 0xFF) / 255.0).toFloat()
                out.b[i] = ((px[o + 2].toInt() and 0xFF) / 255.0).toFloat()
            }
            return out
        }

        /**
         * Converts a [FloatImage] back to 8-bit RGBA exactly like `denoise.to_img`:
         *
         * ```python
         * Image.fromarray((a * 255 + 0.5).astype(np.uint8))
         * ```
         *
         * i.e. multiply by 255, add 0.5, truncate towards zero and clamp to `[0, 255]`
         * (the clamp is a no-op for the reference pipeline, which clips to `[0, 1]` first,
         * but it keeps the function total).
         *
         * @param img       source image, nominally in `[0, 1]`
         * @param alphaFrom optional image whose alpha bytes are copied verbatim into the
         *                  result; when `null` every pixel becomes fully opaque (255).
         *                  The reference tool keeps transparency in a side channel and
         *                  re-attaches it after processing, which is what this models.
         */
        fun toRgba(img: FloatImage, alphaFrom: RgbaImage? = null): RgbaImage {
            val w = img.width
            val h = img.height
            val n = w * h
            val out = ByteArray(n * 4)
            val ar = img.r
            val ag = img.g
            val ab = img.b
            for (i in 0 until n) {
                val o = i * 4
                out[o] = quantize(ar[i])
                out[o + 1] = quantize(ag[i])
                out[o + 2] = quantize(ab[i])
                out[o + 3] = if (alphaFrom != null) alphaFrom.pixels[o + 3] else 0xFF.toByte()
            }
            return RgbaImage(w, h, out)
        }

        /** `(v * 255 + 0.5)` truncated and clamped to a byte, as numpy's `astype(uint8)` does. */
        private fun quantize(v: Float): Byte {
            val q = (v.toDouble() * 255.0 + 0.5).toInt()
            return (if (q < 0) 0 else if (q > 255) 255 else q).toByte()
        }
    }
}
