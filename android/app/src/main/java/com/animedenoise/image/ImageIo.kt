package com.animedenoise.image

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.ColorSpace
import android.graphics.Paint
import android.net.Uri
import android.os.Build
import com.animedenoise.core.RgbaImage
import java.io.InputStream

/** Decoding, encoding and the Bitmap <-> RgbaImage bridge. */
object ImageIo {

    /** 4096x4096, the same ceiling as the desktop web app. */
    const val MAX_PIXELS = 16_777_216

    class Unsupported(message: String) : Exception(message)

    /** Reads bounds first so a huge picture is rejected before the pixels are allocated. */
    fun decodeBounds(resolver: ContentResolver, uri: Uri): Pair<Int, Int> {
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        val stream: java.io.InputStream = try {
            resolver.openInputStream(uri) ?: throw Unsupported(
                "打不开这个文件：如果图片在云相册，请先在相册里下载到手机",
            )
        } catch (e: SecurityException) {
            throw Unsupported("没有访问这个文件的权限，请重新从相册里选择一次")
        }
        stream.use { BitmapFactory.decodeStream(it, null, options) }
        if (options.outWidth <= 0 || options.outHeight <= 0) {
            throw Unsupported("读不出这张图，支持 PNG、JPEG、WebP、BMP、GIF、TIFF、HEIF")
        }
        return options.outWidth to options.outHeight
    }

    /**
     * Decodes to straight RGBA.
     *
     * The bitmap is drawn into an ARGB_8888 surface rather than converted, which is the only way to get a
     * defined pixel format out of hardware bitmaps and 16-bit sources on every OS version we support.
     */
    fun decodeRgba(resolver: ContentResolver, uri: Uri): RgbaImage {
        val bounds = decodeBounds(resolver, uri)
        if (bounds.first * bounds.second > MAX_PIXELS) {
            throw Unsupported(
                "图片太大（" + bounds.first + "×" + bounds.second + "），最多约 1677 万像素"
            )
        }
        if (minOf(bounds.first, bounds.second) < 16) throw Unsupported("图片至少要 16×16")

        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
            inScaled = false
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                inPreferredColorSpace = ColorSpace.get(ColorSpace.Named.SRGB)
            }
        }
        val decoded = resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
            ?: throw Unsupported("打不开这个文件：可能未下载（云图库）或没有访问权限")
        return fromBitmap(decoded)
    }

    /** Copies a Bitmap into the working RGBA layout. Alpha is preserved. */
    fun fromBitmap(bitmap: Bitmap): RgbaImage {
        val w = bitmap.width
        val h = bitmap.height
        val argb = if (bitmap.config == Bitmap.Config.ARGB_8888) {
            bitmap
        } else {
            Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { target ->
                Canvas(target).drawBitmap(bitmap, 0f, 0f, Paint().apply { isFilterBitmap = false })
            }
        }
        val n = w * h
        val pixels = IntArray(n)
        argb.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = ByteArray(n * 4)
        var o = 0
        for (i in 0 until n) {
            val p = pixels[i]
            out[o] = (p ushr 16).toByte()      // R
            out[o + 1] = (p ushr 8).toByte()   // G
            out[o + 2] = p.toByte()            // B
            out[o + 3] = (p ushr 24).toByte()  // A
            o += 4
        }
        if (argb !== bitmap) argb.recycle()
        return RgbaImage(w, h, out)
    }

    /** Builds a Bitmap sharing nothing with [image] but reusing its bytes for the pixel copy. */
    fun toBitmap(image: RgbaImage): Bitmap {
        val n = image.width * image.height
        val pixels = IntArray(n)
        val px = image.pixels
        var o = 0
        for (i in 0 until n) {
            val r = px[o].toInt() and 0xFF
            val g = px[o + 1].toInt() and 0xFF
            val b = px[o + 2].toInt() and 0xFF
            val a = px[o + 3].toInt() and 0xFF
            pixels[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
            o += 4
        }
        return Bitmap.createBitmap(pixels, image.width, image.height, Bitmap.Config.ARGB_8888)
    }

    /**
     * True when any pixel is not fully opaque.
     *
     * The desktop tool only keeps an alpha channel that is actually used (denoise.open_image checks
     * getextrema() != (255, 255)), because an all-opaque RGBA source is just an RGB source and adding
     * an alpha channel would change the output file for no reason. Same rule here.
     */
    fun usesAlpha(image: RgbaImage): Boolean {
        val px = image.pixels
        var i = 3
        while (i < px.size) {
            if (px[i].toInt() and 0xFF != 0xFF) return true
            i += 4
        }
        return false
    }

    /** PNG, lossless, as the desktop tool saves. */
    fun encodePng(image: RgbaImage, quality: Int = 100): ByteArray {
        val bitmap = toBitmap(image)
        return java.io.ByteArrayOutputStream().use { stream ->
            bitmap.compress(Bitmap.CompressFormat.PNG, quality, stream)
            bitmap.recycle()
            stream.toByteArray()
        }
    }

    fun open(resolver: ContentResolver, uri: Uri): InputStream =
        resolver.openInputStream(uri) ?: throw Unsupported("无法打开这个文件")
}
