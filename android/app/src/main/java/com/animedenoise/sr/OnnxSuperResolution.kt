package com.animedenoise.sr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.util.Log
import com.animedenoise.core.RgbaImage
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * ONNX Runtime backend with exact horizontal-band tiling.
 *
 * The network is a pure stack of 3x3 convolutions, so its receptive field is finite. Running a horizontal
 * band with [halo] real neighbour rows above and below produces pixels identical to a whole-image pass,
 * while peak memory stays proportional to one band. [halo] comes from the model itself: the desktop tool
 * derives the same number analytically (denoise.py HEAD_HALO = 96 for the 92-conv 6B trunk), and the
 * exported graph has 96 3x3 convolutions in the trunk plus 4 in the pixel-shuffle tail.
 *
 * Concatenated bands are bit-identical to one whole pass as long as the halo covers the receptive radius.
 */
class OnnxSuperResolution(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    override val scale: Int,
    override val backendName: String,
    private val halo: Int,
    private val bandRows: Int,
) : SuperResolution {

    private val inputName: String
    private val outputName: String
    private var closed = false

    init {
        inputName = session.inputNames.first()
        outputName = session.outputNames.first()
        // Validate the graph output is a float tensor, but do not assume which wrapper class carries it:
        // the desktop jar reports a TensorInfo directly, while the Android AAR reports a NodeInfo that
        // wraps one. Both were observed on real devices, so unwrap defensively instead of trusting one
        // class identity - a strict "is TensorInfo" test here rejected a perfectly good model on the phone.
        val javaType = outputJavaType(session)
        require(javaType == ai.onnxruntime.OnnxJavaType.FLOAT) {
            "model output " + outputName + " is " + javaType + ", expected FLOAT"
        }
    }

    @Synchronized
    override fun upscale(image: RgbaImage): RgbaImage {
        check(!closed) { "session already closed" }
        val w = image.width
        val h = image.height
        val ow = w * scale
        val oh = h * scale
        val out = ByteArray(ow * oh * 4)

        var rows = bandRows
        while (true) {
            try {
                // Banding exists to bound memory, but it costs extra work: each seam recomputes [halo]
                // rows, so a 307-row image split at 192 into two bands evaluates 507 rows (1.65x the
                // compute). Run it whole unless one pass would plausibly exhaust the heap.
                val banding = h.toLong() * w * ESTIMATE_BYTES_PER_PIXEL > BAND_THRESHOLD_BYTES
                if (banding) {
                    runBands(image, out, ow, oh, rows)
                } else {
                    runWhole(image, out, ow, oh)
                }
                return RgbaImage(ow, oh, out)
            } catch (t: Throwable) {
                // Out of memory with big bands: halve and retry. Anything else is a real failure.
                if (rows <= MIN_BAND_ROWS || !looksLikeOom(t)) throw t
                Log.w(TAG, "band of " + rows + " rows failed (" + t.message + "); retrying with " + (rows / 2))
                rows /= 2
            }
        }
    }

    private fun looksLikeOom(t: Throwable): Boolean {
        if (t is OutOfMemoryError) return true
        var c: Throwable? = t
        while (c != null) {
            val m = (c.message ?: "").lowercase()
            if (m.contains("out of memory") || m.contains("bad_alloc") ||
                m.contains("failed to allocate") || m.contains("oom")
            ) return true
            c = c.cause
        }
        return false
    }

    /**
     * One inference over the whole image. Strictly less work than banding, so it is preferred whenever
     * the intermediate tensors fit: a 64-channel trunk at 307x307 is roughly 307*307*64*4 bytes = 24 MB.
     */
    private fun runWhole(image: RgbaImage, out: ByteArray, ow: Int, oh: Int) {
        val w = image.width
        val h = image.height
        val flat = preprocess(image, 0, h)
        OnnxTensor.createTensor(env, flat, longArrayOf(1, 3, h.toLong(), w.toLong())).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                val node = requireTensor(rawOutput(result, outputName), outputName)
                @Suppress("UNCHECKED_CAST")
                val value = node.value as Array<Array<Array<FloatArray>>>
                writeBand(
                    planes = value[0],
                    ow = ow,
                    oh = oh,
                    destRowStart = 0,
                    srcRowStart = 0,
                    rows = h * scale,
                    out = out,
                )
            }
        }
    }

    private fun runBands(image: RgbaImage, out: ByteArray, ow: Int, oh: Int, rows: Int) {
        val w = image.width
        val h = image.height
        var r0 = 0
        while (r0 < h) {
            val r1 = minOf(h, r0 + rows)
            val a = maxOf(0, r0 - halo)
            val b = minOf(h, r1 + halo)
            val bandH = b - a

            val flat = preprocess(image, a, bandH)
            OnnxTensor.createTensor(env, flat, longArrayOf(1, 3, bandH.toLong(), w.toLong())).use { tensor ->
                session.run(mapOf(inputName to tensor)).use { result ->
                    val node = requireTensor(rawOutput(result, outputName), outputName)
                    @Suppress("UNCHECKED_CAST")
                    val value = node.value as Array<Array<Array<FloatArray>>>
                    writeBand(
                        planes = value[0],
                        ow = ow,
                        oh = oh,
                        destRowStart = r0 * scale,
                        srcRowStart = (r0 - a) * scale,
                        rows = (r1 - r0) * scale,
                        out = out,
                    )
                }
            }
            r0 = r1
        }
    }

    /** Interleaved RGBA bytes -> [3, bandH, w] planar float in [0,1], NCHW-friendly contiguous layout. */
    private fun preprocess(image: RgbaImage, startRow: Int, rows: Int): java.nio.FloatBuffer {
        val w = image.width
        val px = image.pixels
        val plane = rows * w
        val buf = ByteBuffer.allocateDirect(3 * plane * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        var idx = startRow * w * 4
        var o = 0
        for (r in 0 until rows) {
            for (c in 0 until w) {
                val p = r * w + c
                val red = px[idx].toInt() and 0xFF
                val green = px[idx + 1].toInt() and 0xFF
                val blue = px[idx + 2].toInt() and 0xFF
                buf.put(p, red * INV_255)
                buf.put(plane + p, green * INV_255)
                buf.put(2 * plane + p, blue * INV_255)
                idx += 4
                o++
            }
        }
        buf.position(0)
        return buf
    }

    /** [3][bandOutH][ow] planar float -> interleaved RGBA bytes with Python's (v*255+0.5) truncation. */
    /**
     * Copies [rows] output rows out of one band.
     *
     * Two different offsets are involved and mixing them up silently shifts everything below the first
     * band: the *destination* row is fixed by where the band starts in the output image
     * ([destRowStart] = firstInputRow * scale), while the *source* row inside the band's own tensor is
     * shifted by the halo rows the band carries above it ([srcRowStart] = (firstInputRow - haloStart) *
     * scale).
     */
    private fun writeBand(
        planes: Array<Array<FloatArray>>,
        ow: Int,
        oh: Int,
        destRowStart: Int,
        srcRowStart: Int,
        rows: Int,
        out: ByteArray,
    ) {
        for (r in 0 until rows) {
            val y = destRowStart + r
            require(y < oh) { "band wrote row " + y + " past " + oh }
            var o = (y * ow) * 4
            val srcRow = srcRowStart + r
            val pr = planes[0][srcRow]
            val pg = planes[1][srcRow]
            val pb = planes[2][srcRow]
            for (c in 0 until ow) {
                out[o] = clamp8(pr[c])
                out[o + 1] = clamp8(pg[c])
                out[o + 2] = clamp8(pb[c])
                out[o + 3] = -1 // 255
                o += 4
            }
        }
    }

    private fun clamp8(v: Float): Byte {
        val i = (v * 255f + 0.5f).toInt()
        return (if (i < 0) 0 else if (i > 255) 255 else i).toByte()
    }

    /**
     * Unwraps whatever the runtime's Result container hands back for [name].
     *
     * Established on a Redmi Note 12 Turbo: the Android AAR returns java.util.Optional wrapping the
     * OnnxValue, whereas the desktop jar returns the OnnxValue itself. Guessing one shape silently broke
     * every backend on-device while the JVM tests kept passing, so both are handled explicitly.
     */
    private fun rawOutput(result: OrtSession.Result, name: String): Any? {
        val raw: Any? = result.get(name)
        return if (raw is java.util.Optional<*>) raw.orElse(null) else raw
    }

    private fun requireTensor(value: Any?, name: String): OnnxTensor =
        value as? OnnxTensor
            ?: error("output " + name + " is " + (value?.javaClass?.name ?: "absent") + ", expected a tensor")

    /** The Java type of the graph's single output, whatever wrapper the runtime uses to describe it. */
    private fun outputJavaType(session: OrtSession): ai.onnxruntime.OnnxJavaType? {
        val info = session.outputInfo[outputName] ?: return null
        if (info is TensorInfo) return info.type
        // Android AAR: NodeInfo(name, info=TensorInfo(...)).
        return try {
            val field = info.javaClass.getDeclaredField("info").apply { isAccessible = true }
            (field.get(info) as? TensorInfo)?.type
        } catch (t: Throwable) {
            null
        }
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        session.close()
    }

    companion object {
        private const val TAG = "OnnxSR"
        private val OnnxTensorType = ai.onnxruntime.OnnxJavaType.FLOAT

        /** Below this, banding overhead dominates and an OOM means something else is wrong. */
        const val MIN_BAND_ROWS = 32

        /** Rough per-input-pixel cost of one whole pass, in bytes (64-channel trunk + tail buffers). */
        const val ESTIMATE_BYTES_PER_PIXEL = 400L

        /** Past this the whole-image pass is risky on a phone, so bands are used to bound the peak. */
        const val BAND_THRESHOLD_BYTES = 320L * 1024 * 1024

        /** 1/255 exactly matches numpy's (px / 255) up to float32 rounding. */
        private const val INV_255 = 1f / 255f
    }
}
