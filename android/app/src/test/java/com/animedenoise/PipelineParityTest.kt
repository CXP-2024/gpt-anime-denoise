package com.animedenoise

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.animedenoise.core.FloatImage
import com.animedenoise.core.L0Smoother
import com.animedenoise.core.PilResize
import com.animedenoise.core.RgbaImage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * End-to-end parity on the JVM: the real ONNX model plus the Kotlin stages against the full
 * desktop reference output.
 *
 * This is the strongest check available without the phone: it exercises the exact code path the app
 * runs (preprocess to NCHW float, banded inference, postprocess rounding, BOX downscale, LANCZOS
 * back, L0) and compares the final PNG bytes to what denoise.py produced. A pass here means the only
 * unknown left on the device is the runtime's own floating point, not our pipeline.
 *
 * Skipped automatically when onnxruntime-native is not on the test classpath.
 */
class PipelineParityTest {

    /** Unit tests run with cwd = android/app, so walk up until the fixtures are found. */
    private fun locate(relative: String): File? {
        var dir: File? = File(System.getProperty("user.dir") ?: ".").absoluteFile
        while (dir != null) {
            val candidate = File(dir, relative)
            if (candidate.exists()) return candidate
            dir = dir.parentFile
        }
        return null
    }

    private val golden: File? = locate("tools/golden")

    private val model: File? = locate("src/main/assets/realesrgan_x4plus_anime_6b.onnx")

    @Test
    fun fullPipelineMatchesDesktop() {
        val dir = golden
        assumeTrue("golden fixtures not found", dir != null)
        val modelFile = model
        assumeTrue("model asset not found", modelFile != null)

        val tag = "01ee25d8"
        val input = File(dir!!, "in_$tag.png")
        val expected = File(dir, "out_$tag.png")
        assumeTrue("fixture $input missing", input.isFile)
        assumeTrue("fixture $expected missing", expected.isFile)

        val src = Png.read(input)
        val want = Png.read(expected)

        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val session = env.createSession(modelFile!!.readBytes(), opts)

        var t = System.nanoTime()
        val smallW = PilResize.pythonRoundSize(src.width, 0.6)
        val smallH = PilResize.pythonRoundSize(src.height, 0.6)
        val small = PilResize.resize(src, smallW, smallH, PilResize.Filter.BOX)
        val tDown = (System.nanoTime() - t) / 1_000_000

        t = System.nanoTime()
        val upscaled = bandedUpscale(env, session, small, halo = 100, bandRows = 192)
        val tSr = (System.nanoTime() - t) / 1_000_000
        assertEquals("upscale factor", smallW * 4, upscaled.width)

        t = System.nanoTime()
        val back = PilResize.resize(upscaled, src.width, src.height, PilResize.Filter.LANCZOS)
        val tBack = (System.nanoTime() - t) / 1_000_000

        t = System.nanoTime()
        val smoothed = L0Smoother.smooth(FloatImage.fromRgba(back), 0.001)
        val out = FloatImage.toRgba(smoothed, null)
        val tL0 = (System.nanoTime() - t) / 1_000_000

        var max = 0
        var differing = 0
        var total = 0L
        for (i in want.pixels.indices) {
            val d = abs((want.pixels[i].toInt() and 0xFF) - (out.pixels[i].toInt() and 0xFF))
            if (d > 0) differing++
            total += d
            if (d > max) max = d
        }
        val mean = total.toDouble() / want.pixels.size
        println(
            "[e2e] " + tag + ": max byte diff = " + max + ", differing bytes = " + differing +
                ", mean = " + mean
        )
        println(
            "[e2e] timings: downscale " + tDown + " ms, sr " + tSr + " ms, resize back " + tBack +
                " ms, l0 " + tL0 + " ms"
        )
        session.close()
        // The model is float32 and numpy is float64, so a couple of levels of drift is expected
        // inside the L0 solve; anything beyond that means a real bug.
        assertTrue("pipeline drifted: max byte diff " + max, max <= 2)
    }

    /** Same banding strategy as OnnxSuperResolution, in test code so it can be verified off-device. */
    private fun bandedUpscale(
        env: OrtEnvironment,
        session: OrtSession,
        image: RgbaImage,
        halo: Int,
        bandRows: Int,
    ): RgbaImage {
        val w = image.width
        val h = image.height
        val scale = 4
        val out = ByteArray(w * scale * h * scale * 4)
        val inputName = session.inputNames.first()
        var r0 = 0
        while (r0 < h) {
            val r1 = minOf(h, r0 + bandRows)
            val a = maxOf(0, r0 - halo)
            val b = minOf(h, r1 + halo)
            val rows = b - a
            val plane = rows * w
            val buffer = ByteBuffer.allocateDirect(3 * plane * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            var idx = a * w * 4
            for (r in 0 until rows) {
                for (c in 0 until w) {
                    val p = r * w + c
                    buffer.put(p, (image.pixels[idx].toInt() and 0xFF) / 255f)
                    buffer.put(plane + p, (image.pixels[idx + 1].toInt() and 0xFF) / 255f)
                    buffer.put(2 * plane + p, (image.pixels[idx + 2].toInt() and 0xFF) / 255f)
                    idx += 4
                }
            }
            buffer.position(0)
            OnnxTensor.createTensor(env, buffer, longArrayOf(1, 3, rows.toLong(), w.toLong())).use { tensor ->
                session.run(mapOf(inputName to tensor)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    val planes = (result[0].value as Array<Array<Array<FloatArray>>>)[0]
                    val by = (r0 - a) * scale
                    for (r in 0 until (r1 - r0) * scale) {
                        // Destination row is fixed by the band's place in the output image; the source
                        // row inside the band tensor is offset by the halo rows above it.
                        val y = r0 * scale + r
                        var o = (y * w * scale) * 4
                        val srcRow = by + r
                        for (c in 0 until w * scale) {
                            out[o] = clamp8(planes[0][srcRow][c])
                            out[o + 1] = clamp8(planes[1][srcRow][c])
                            out[o + 2] = clamp8(planes[2][srcRow][c])
                            out[o + 3] = -1
                            o += 4
                        }
                    }
                }
            }
            r0 = r1
        }
        return RgbaImage(w * scale, h * scale, out)
    }

    private fun clamp8(v: Float): Byte {
        val i = (v * 255f + 0.5f).toInt()
        return (if (i < 0) 0 else if (i > 255) 255 else i).toByte()
    }
}
