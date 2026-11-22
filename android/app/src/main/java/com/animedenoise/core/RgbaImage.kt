package com.animedenoise.core

/**
 * 8-bit RGBA image in Android ARGB_8888 byte order: R,G,B,A per pixel, row-major.
 *
 * Same layout as Bitmap.Config.ARGB_8888 memory, so a Bitmap round-trip is a plain copy.
 */
class RgbaImage(val width: Int, val height: Int, val pixels: ByteArray) {
    init {
        require(width > 0 && height > 0) { "empty image: " + width + "x" + height }
        require(pixels.size == width * height * 4) {
            "pixel buffer is " + pixels.size + " bytes, expected " + (width * height * 4) +
                " for " + width + "x" + height
        }
    }

    override fun toString(): String = "RgbaImage(" + width + "x" + height + ")"
}
