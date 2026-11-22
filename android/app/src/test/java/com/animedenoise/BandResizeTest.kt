package com.animedenoise

import com.animedenoise.core.PilResize
import com.animedenoise.core.RgbaImage
import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File
import kotlin.math.abs

/** A resize computed strip by strip must equal the corresponding rows of a whole-image resize. */
class BandResizeTest {

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
    fun stripsMatchWholeImageResize() {
        val golden = locate("tools/golden") ?: return
        val file = File(golden, "in_01ee25d8.png")
        if (!file.isFile) return
        val src = Png.read(file)

        for (filter in listOf(PilResize.Filter.BOX, PilResize.Filter.LANCZOS)) {
            val dstW = PilResize.pythonRoundSize(src.width, 0.6)
            val dstH = PilResize.pythonRoundSize(src.height, 0.6)
            val whole = PilResize.resize(src, dstW, dstH, filter)

            var row = 0
            var worst = 0
            while (row < dstH) {
                val end = minOf(dstH, row + 37)  // deliberately not aligned to anything
                val band = PilResize.resizeRowBand(src, dstW, dstH, row, end, filter)
                assertEquals("band height", end - row, band.height)
                for (y in 0 until band.height) {
                    val wRow = (row + y) * dstW
                    val bRow = y * dstW
                    for (x in 0 until dstW) {
                        for (c in 0 until 4) {
                            val a = whole.pixels[(wRow + x) * 4 + c].toInt() and 0xFF
                            val b = band.pixels[(bRow + x) * 4 + c].toInt() and 0xFF
                            val d = abs(a - b)
                            if (d > worst) worst = d
                        }
                    }
                }
                row = end
            }
            println("[band] " + filter + ": worst byte diff vs whole-image resize = " + worst)
            assertEquals("band resize must be exact for " + filter, 0, worst)
        }
    }
}
