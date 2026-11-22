package com.animedenoise

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.animedenoise.core.PilResize
import com.animedenoise.core.RgbaImage
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/** Checks the ONNX plumbing itself: does a whole-image pass from Kotlin reproduce the Python tensor? */
class OnnxDiagTest {

    private fun locate(relative: String): File? {
        var dir: File? = File(System.getProperty("user.dir") ?: ".").absoluteFile
        while (dir != null) {
            val candidate = File(dir, relative)
            if (candidate.exists()) return candidate
            dir = dir.parentFile
        }
        return null
    }

    @Test
    fun wholeImageMatchesPython() {
        val golden = locate("tools/golden") ?: return
        val modelFile = locate("src/main/assets/realesrgan_x4plus_anime_6b.onnx") ?: return
        val src = Png.read(File(golden, "in_01ee25d8.png"))
        val sw = PilResize.pythonRoundSize(src.width, 0.6)
        val sh = PilResize.pythonRoundSize(src.height, 0.6)
        val small = PilResize.resize(src, sw, sh, PilResize.Filter.BOX)

        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val session = env.createSession(modelFile.readBytes(), opts)
        println("[onnx] inputs=" + session.inputNames + " outputs=" + session.outputNames)

        val w = small.width
        val h = small.height
        val plane = w * h
        val buffer = ByteBuffer.allocateDirect(3 * plane * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        for (p in 0 until plane) {
            val o = p * 4
            buffer.put(p, (small.pixels[o].toInt() and 0xFF) / 255f)
            buffer.put(plane + p, (small.pixels[o + 1].toInt() and 0xFF) / 255f)
            buffer.put(2 * plane + p, (small.pixels[o + 2].toInt() and 0xFF) / 255f)
        }
        buffer.position(0)
        OnnxTensor.createTensor(env, buffer, longArrayOf(1, 3, h.toLong(), w.toLong())).use { tensor ->
            session.run(mapOf(session.inputNames.first() to tensor)).use { result ->
                val out = result[0]
                println("[onnx] output tensor type=" + out.javaClass.simpleName + " info=" + out.info)
                @Suppress("UNCHECKED_CAST")
                val planes = (out.value as Array<Array<Array<FloatArray>>>)[0]
                println("[onnx] planes=" + planes.size + " rows=" + planes[0].size + " cols=" + planes[0][0].size)
                // Rebuild the 4x image and compare against the Python-derived mid fixture after resize.
                val bw = w * 4
                val bh = h * 4
                val out4 = ByteArray(bw * bh * 4)
                var o = 0
                for (c in 0 until 3) {
                    val pl = planes[c]
                    for (y in 0 until bh) {
                        for (x in 0 until bw) {
                            val v = (pl[y][x] * 255f + 0.5f).toInt().coerceIn(0, 255)
                            out4[((y * bw + x) * 4) + c] = v.toByte()
                        }
                    }
                }
                for (i in 0 until bw * bh) out4[i * 4 + 3] = -1
                val up = RgbaImage(bw, bh, out4)
                val back = PilResize.resize(up, 512, 512, PilResize.Filter.LANCZOS)
                val mid = Npy.readFloatImage(File(golden, "mid_01ee25d8.npy"))
                var max = 0
                var sum = 0L
                for (i in 0 until 512 * 512) {
                    val got = back.pixels[i * 4].toInt() and 0xFF
                    val want = Math.round(mid.r[i] * 255f).toInt().coerceIn(0, 255)
                    val d = abs(got - want)
                    if (d > max) max = d
                    sum += d
                }
                println("[onnx] 4x -> resize back vs mid R channel: max=" + max + " mean=" + (sum.toDouble() / (512 * 512)))
            }
        }
        session.close()
    }
}
