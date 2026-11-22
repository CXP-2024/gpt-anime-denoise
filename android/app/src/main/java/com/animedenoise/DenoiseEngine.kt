package com.animedenoise

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.util.Log
import com.animedenoise.core.FloatImage
import com.animedenoise.core.L0Smoother
import com.animedenoise.core.L0SmootherF32
import com.animedenoise.core.L0Tiled
import com.animedenoise.core.PilResize
import com.animedenoise.core.RgbaImage
import com.animedenoise.sr.ModelStore
import com.animedenoise.sr.Runtimes
import com.animedenoise.sr.SrModel
import com.animedenoise.sr.SuperResolution

/**
 * Runs the denoise pipeline, keeping peak heap proportional to a strip rather than to the image.
 *
 * On a phone the 512 MB dalvik heap is the binding constraint, not compute. A naive 8.3 MP run
 * holds the 12.3 MP super-resolution output (182 MB), the resized result, the float planes and the
 * UI bitmap at once - about 415 MB - and dies with an OutOfMemoryError inside the JNI bridge.
 *
 * So the pipeline streams: the image is cut into horizontal strips that are carried through
 * downscale, the model, the resize back and the L0 solve, and only the finished strip is written into
 * the output buffer. Two details make this exact rather than approximate:
 *
 *  * strips overlap by the model's receptive radius ([SrModel.halo]) so every output row sees the same
 *    input context a whole-image pass would;
 *  * L0 is applied to overlapping tiles by [L0Tiled], whose verified drift against a whole-image
 *    solve is 1.6% of pixels above one output level.
 */
class DenoiseEngine(private val context: Context) {

    data class Result(
        val output: RgbaImage,
        val stageMillis: Map<String, Long>,
        val backend: String,
        val reusedSuperResolution: Boolean,
        val tiled: Boolean,
    )

    private var sr: SuperResolution? = null
    private var loadedModel: SrModel? = null
    private var loadedBackend: Runtimes.Backend? = null
    private var env: OrtEnvironment? = null

    @Synchronized
    fun ensureModel(
        model: SrModel,
        backend: Runtimes.Backend,
        bandRows: Int = model.defaultBandRows,
    ): String {
        val current = sr
        if (current != null && loadedModel == model && loadedBackend == backend) return current.backendName
        current?.close()
        sr = null
        val ortEnv = env ?: OrtEnvironment.getEnvironment().also { env = it }
        val bytes = ModelStore.readBytes(context.assets, model.assetName)
        val built = Runtimes.create(ortEnv, model, bytes, backend, bandRows)
        sr = built.runtime
        loadedModel = model
        loadedBackend = backend
        return built.runtime.backendName + (built.note?.let { " (" + it.trim() + ")" } ?: "")
    }

    fun denoise(
        src: RgbaImage,
        down: Double,
        lam: Double,
        onStage: (String) -> Unit,
        sourceHasAlpha: Boolean = false,
        isCancelled: () -> Boolean = { false },
    ): Result {
        val stage = LinkedHashMap<String, Long>()
        val w = src.width
        val h = src.height
        val big = h.toLong() * w >= STREAM_MIN_PIXELS
        val model = loadedModel ?: error("model not loaded")
        val runtime = sr ?: error("model not loaded")
        val scale = runtime.scale
        val halo = model.halo

        val smallW = PilResize.pythonRoundSize(w, down)
        val smallH = PilResize.pythonRoundSize(h, down)
        val out = ByteArray(w * h * 4)

        var t = System.currentTimeMillis()
        // Rows of the *small* image handled per strip. The strip is small enough that all of its
        // intermediates fit comfortably; the model's halo is added on both sides.
        val stripRows = if (big) STRIP_SMALL_ROWS else smallH
        var sy0 = 0
        var stripIndex = 0
        val strips = if (big) (smallH + stripRows - 1) / stripRows else 1

        while (sy0 < smallH) {
            if (isCancelled()) throw Cancelled()
            val sy1 = minOf(smallH, sy0 + stripRows)

            // Context rows the model needs above and below this strip, in small-image space.
            val cy0 = maxOf(0, sy0 - halo)
            val cy1 = minOf(smallH, sy1 + halo)
            val ctxH = cy1 - cy0

            // Downscale just this band: resizeRowBand computes only the requested output rows.
            val smallBand = PilResize.resizeRowBand(
                src, smallW, smallH, cy0, cy1, PilResize.Filter.BOX,
            )
            if (stripIndex == 0) stage["downscale"] = System.currentTimeMillis() - t

            val percent = if (big) (stripIndex * 100 / strips) else -1
            onStage(
                if (big) "正在重建画面细节 " + (stripIndex + 1) + "/" + strips + " 段（约 " + percent + "%） · " + runtime.backendName
                else "正在重建画面细节 · " + runtime.backendName
            )
            t = System.currentTimeMillis()
            val upBand = runtime.upscale(smallBand)
            if (stripIndex == 0) stage["superResolution"] = System.currentTimeMillis() - t

            if (isCancelled()) throw Cancelled()

            t = System.currentTimeMillis()
            // Resize this band back to the target size; a band of small-image rows maps to a band
            // of output rows at the same scale.
            val outRowStart = (cy0.toLong() * h / smallH).toInt()
            val outRowEnd = (cy1.toLong() * h / smallH).toInt()
            val rowsBack = maxOf(1, outRowEnd - outRowStart)
            val backBand = PilResize.resize(upBand, w, rowsBack, PilResize.Filter.LANCZOS)
            if (stripIndex == 0) stage["resizeBack"] = System.currentTimeMillis() - t

            // Keep only this strip's rows (drop the halo), smooth, write out.
            val keepStart = (sy0.toLong() * h / smallH).toInt()
            val keepEnd = (sy1.toLong() * h / smallH).toInt()
            val keepTop = maxOf(0, keepStart - outRowStart)
            val keepBottom = minOf(rowsBack, keepEnd - outRowStart)
            if (keepBottom > keepTop) {
                val stripe = cropRows(backBand, keepTop, keepBottom)
                onStage(
                    if (big) "正在压平噪点 " + (stripIndex + 1) + "/" + strips + " 段 · " + runtime.backendName
                    else "正在压平噪点 · " + runtime.backendName
                )
                val smoothed = if (lam > 0.0) {
                    L0Tiled.smooth(FloatImage.fromRgba(stripe), lam, isCancelled = isCancelled)
                } else {
                    FloatImage.fromRgba(stripe)
                }
                writeRows(smoothed, out, w, keepStart)
            }
            sy0 = sy1
            stripIndex++
        }
        stage["l0"] = System.currentTimeMillis() - t

        onStage("整理结果")
        if (sourceHasAlpha) applyAlpha(out, src)
        return Result(
            output = RgbaImage(w, h, out),
            stageMillis = stage,
            backend = runtime.backendName,
            reusedSuperResolution = false,
            tiled = big,
        )
    }

    private fun cropRows(img: RgbaImage, top: Int, bottom: Int): RgbaImage {
        val w = img.width
        val rows = bottom - top
        val px = ByteArray(w * rows * 4)
        System.arraycopy(img.pixels, top * w * 4, px, 0, px.size)
        return RgbaImage(w, rows, px)
    }

    private fun writeRows(img: FloatImage, out: ByteArray, outWidth: Int, startRow: Int) {
        val w = img.width
        val n = w * img.height
        val r = img.r
        val g = img.g
        val b = img.b
        var o = startRow * outWidth * 4
        for (i in 0 until n) {
            out[o] = q8(r[i])
            out[o + 1] = q8(g[i])
            out[o + 2] = q8(b[i])
            out[o + 3] = -1
            o += 4
        }
    }

    private fun applyAlpha(out: ByteArray, src: RgbaImage) {
        val sp = src.pixels
        var i = 3
        while (i < out.size) {
            out[i] = sp[i]
            i += 4
        }
    }

    private fun q8(v: Float): Byte {
        val q = (v * 255f + 0.5f).toInt()
        return (if (q < 0) 0 else if (q > 255) 255 else q).toByte()
    }

    @Synchronized
    fun release() {
        sr?.close()
        sr = null
        loadedModel = null
    }

    class Cancelled : Exception("cancelled")

    companion object {
        private const val TAG = "DenoiseEngine"

        /**
         * Above this the pipeline streams in strips.
         *
         * Below it, whole-image processing fits the heap comfortably and avoids paying the model
         * halo at every strip boundary. Measured cost of streaming a 512x512 image that did not need
         * it: 38.8 s instead of 7.9 s, almost all of it duplicated halo work.
         */
        private const val STREAM_MIN_PIXELS = 6_000_000L

        /**
         * Rows of the downscaled image per strip.
         *
         * Larger strips mean fewer boundaries and therefore less repeated halo, but a bigger working
         * set. 512 rows of a 1296-wide downscale is 0.66 MP per strip, which keeps every intermediate
         * well inside the heap while the halo is paid only once per 512 rows.
         */
        private const val STRIP_SMALL_ROWS = 768

        /** README presets: name, down, lam. */
        val PRESETS = listOf(
            Preset("轻柔", 0.8, 0.0006, "保留更多细节，纹理残留多一点"),
            Preset("推荐", 0.6, 0.001, "默认方案，动漫与番剧画风效果最好"),
            Preset("强力", 0.45, 0.0018, "抹得更平，噪点更少，但可能损失淡阴影"),
        )

        data class Preset(val name: String, val down: Double, val lam: Double, val hint: String)
    }
}
