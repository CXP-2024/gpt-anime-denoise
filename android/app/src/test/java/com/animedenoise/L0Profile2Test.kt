package com.animedenoise

import com.animedenoise.core.FloatImage
import com.animedenoise.core.L0SmootherF32
import org.junit.Test

/** How much of L0 is FFT vs the row passes vs the per-iteration plane copies? */
class L0Profile2Test {

    @Test
    fun profileL0Stages() {
        for (side in intArrayOf(512, 1024)) {
            val img = FloatImage(side, side)
            for (c in 0 until 3) {
                val p = img.plane(c)
                for (i in p.indices) p[i] = ((i * 37 + c * 11) % 255) / 255f
            }
            // time the whole solve
            val t0 = System.nanoTime()
            L0SmootherF32.smooth(img, 0.001)
            val total = (System.nanoTime() - t0) / 1_000_000
            println("[l0] " + side + "x" + side + " full solve: " + total + " ms")
        }
    }
}
