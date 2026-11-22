package com.animedenoise

import com.animedenoise.core.Fft2DF32
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin

/** Isolates the cost of a 2D FFT pass so L0 optimisation targets the real hot spot. */
class L0ProfileTest {

    @Test
    fun profileFftVsElementwise() {
        for (side in intArrayOf(256, 512, 1024)) {
            val n = side * side
            val re = Array(3) { FloatArray(n) { i -> (i % 251) / 251f } }
            val im = Array(3) { FloatArray(n) }
            val fft = Fft2DF32(side, side)

            // warm up
            fft.transformChannels(re, im, false)

            val t0 = System.nanoTime()
            repeat(3) { fft.transformChannels(re, im, false) }
            val fftMs = (System.nanoTime() - t0) / 1_000_000 / 3

            // elementwise pass of the same shape, for comparison
            val out = Array(3) { FloatArray(n) }
            val k = FloatArray(n) { 0.5f }
            val b = 3.5f
            val t1 = System.nanoTime()
            repeat(3) {
                for (c in 0 until 3) {
                    val o = out[c]
                    val r = re[c]
                    val m = im[c]
                    for (i in 0 until n) o[i] = (r[i] + b * m[i]) * k[i]
                }
            }
            val elemMs = (System.nanoTime() - t1) / 1_000_000 / 3

            println("[prof] " + side + "x" + side + " (" + (n / 1e6) + " MP): FFT " + fftMs +
                " ms, elementwise " + elemMs + " ms, ratio " + String.format(java.util.Locale.US, "%.1f", fftMs.toDouble() / elemMs.coerceAtLeast(1)))
        }
    }
}
