package com.animedenoise

import com.animedenoise.core.Fft2DF32
import org.junit.Test

/**
 * Splits one beta iteration into its four components so optimisation targets the real cost.
 */
class L0Profile3Test {

    @Test
    fun splitIterationCost() {
        val side = 1024
        val n = side * side
        val h = side
        val w = side
        val s = Array(3) { FloatArray(n) { i -> (i % 251) / 251f } }
        val hDiff = Array(3) { FloatArray(n) }
        val vDiff = Array(3) { FloatArray(n) }
        val workRe = Array(3) { FloatArray(n) }
        val workIm = Array(3) { FloatArray(n) }
        val fft = Fft2DF32(h, w)

        // 1) FFT cost, one channel set
        var t = System.nanoTime()
        fft.transformChannels(workRe, workIm, false)
        val fftMs = (System.nanoTime() - t) / 1_000_000

        // 2) diff + mask + divergence
        t = System.nanoTime()
        for (c in 0 until 3) {
            val sc = s[c]; val hd = hDiff[c]; val vd = vDiff[c]
            for (y in 0 until h) {
                val base = y * w
                val last = base + w - 1
                for (x in base until last) hd[x] = sc[x + 1] - sc[x]
                hd[last] = sc[base] - sc[last]
                val next = if (y == h - 1) 0 else base + w
                for (x in 0 until w) vd[base + x] = sc[next + x] - sc[base + x]
            }
        }
        val diffMs = (System.nanoTime() - t) / 1_000_000

        // 3) mask pass
        t = System.nanoTime()
        val threshold = 1e-6f
        for (y in 0 until h) {
            val base = y * w
            for (x in 0 until w) {
                val i = base + x
                val e = hDiff[0][i] * hDiff[0][i] + vDiff[0][i] * vDiff[0][i]
                if (e < threshold) { hDiff[0][i] = 0f; vDiff[0][i] = 0f }
            }
        }
        val maskMs = (System.nanoTime() - t) / 1_000_000

        // 4) copy loop
        t = System.nanoTime()
        for (c in 0 until 3) {
            val sc = s[c]; val wc = workRe[c]
            for (i in 0 until n) sc[i] = wc[i]
        }
        val copyMs = (System.nanoTime() - t) / 1_000_000

        println("[split] " + side + "x" + side + ": fft=" + fftMs + " diff=" + diffMs +
            " mask=" + maskMs + " copy=" + copyMs + " (ms per iteration component)")
    }
}
