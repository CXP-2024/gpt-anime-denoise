package com.animedenoise

import com.animedenoise.core.FloatImage
import com.animedenoise.core.L0Smoother
import com.animedenoise.core.PilResize
import com.animedenoise.core.RgbaImage
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * Narrowing diagnostics: compares each Kotlin stage against the corresponding Python intermediate
 * so a drift can be attributed to one stage instead of the whole pipeline.
 */
class StageDiagTest {

    private fun locate(relative: String): File? {
        var dir: File? = File(System.getProperty("user.dir") ?: ".").absoluteFile
        while (dir != null) {
            val candidate = File(dir, relative)
            if (candidate.exists()) return candidate
            dir = dir.parentFile
        }
        return null
    }

    private val golden: File = locate("tools/golden") ?: File("tools/golden")

    @Test
    fun compareStages() {
        val src = Png.read(File(golden, "in_01ee25d8.png"))
        val mid = Npy.readFloatImage(File(golden, "mid_01ee25d8.npy"))

        // 1) BOX downscale against Pillow
        val sw = PilResize.pythonRoundSize(src.width, 0.6)
        val sh = PilResize.pythonRoundSize(src.height, 0.6)
        val boxed = PilResize.resize(src, sw, sh, PilResize.Filter.BOX)
        val wantBox = Png.read(File(golden, "resize_boxed.png"))
        println("[diag] BOX: " + maxDiff(boxed, wantBox))

        // 2) If we feed the *known-good* 4x image, does LANCZOS back reproduce mid exactly?
        //    Build the 4x image from mid is impossible; instead check LANCZOS on the boxed fixture.
        val backDrop = PilResize.resize(Png.read(File(golden, "resize_boxed.png")), src.width, src.height, PilResize.Filter.LANCZOS)
        val wantBack = Png.read(File(golden, "resize_back.png"))
        println("[diag] LANCZOS: " + maxDiff(backDrop, wantBack))

        // 3) L0 on the reference mid: should reproduce out_*.png
        val smoothed = L0Smoother.smooth(mid, 0.001)
        val got = FloatImage.toRgba(smoothed, backDrop)
        val wantOut = Png.read(File(golden, "out_01ee25d8.png"))
        println("[diag] L0 from mid: " + maxDiff(got, wantOut))

        // 4) resized-back (from boxed, ignoring the model) vs mid: shows how far the model stage would
        //    legitimately move pixels; a correct model makes this small.
        var max = 0.0
        var total = 0.0
        for (i in 0 until mid.r.size) {
            val v = backDrop.pixels[i * 4].toInt() and 0xFF
            val d = abs(v / 255.0 - mid.r[i])
            if (d > max) max = d
            total += d
        }
        println("[diag] no-model back vs mid: max=" + max + " mean=" + (total / mid.r.size))
    }

    @Test
    fun l0Sanity() {
        // A flat image must survive L0 untouched (no NaN, no shift); catches FFT scaling errors.
        val img = RgbaImage(64, 64, ByteArray(64 * 64 * 4) { 0x40 })
        val flat = FloatImage.fromRgba(img)
        val out = L0Smoother.smooth(flat, 0.001)
        var max = 0f
        for (i in 0 until out.r.size) {
            max = maxOf(max, abs(out.r[i] - 0.2509804f), abs(out.g[i] - 0.2509804f), abs(out.b[i] - 0.2509804f))
        }
        println("[diag] flat image L0 delta = " + max)
    }

    private fun maxDiff(a: RgbaImage, b: RgbaImage): String {
        if (a.width != b.width || a.height != b.height) {
            return "SIZE " + a.width + "x" + a.height + " vs " + b.width + "x" + b.height
        }
        var max = 0
        var differing = 0
        for (i in a.pixels.indices) {
            val d = abs((a.pixels[i].toInt() and 0xFF) - (b.pixels[i].toInt() and 0xFF))
            if (d > 0) differing++
            if (d > max) max = d
        }
        return "max=" + max + " differing=" + differing
    }
}
