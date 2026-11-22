package com.animedenoise.core

/** Row/column 2D FFT over float planes, batched across the three colour channels. */
class Fft2DF32(val height: Int, val width: Int) {

    private val rowFft = Fft1DF32(width)
    private val columnFft = Fft1DF32(height)
    private val columnScratch = ThreadLocal.withInitial {
        FloatPair(FloatArray(height), FloatArray(height))
    }

    fun transformChannels(re: Array<FloatArray>, im: Array<FloatArray>, inverse: Boolean) {
        val n = height * width
        for (c in re.indices) {
            require(re[c].size >= n && im[c].size >= n) {
                "plane " + c + " is smaller than " + width + "x" + height
            }
        }
        // Rows are contiguous, so the 1-D FFT runs in place on a slice of the plane.
        Parallel.forEachRange(re.size * height) { from, to ->
            var idx = from
            while (idx < to) {
                val c = idx / height
                val y = idx % height
                rowFft.transform(re[c], im[c], y * width, inverse)
                idx++
            }
        }
        // Columns are strided: gather, transform, scatter.
        Parallel.forEachRange(re.size * width) { from, to ->
            var idx = from
            while (idx < to) {
                val c = idx / width
                val x = idx % width
                val sc = columnScratch.get()
                val rc = re[c]
                val ic = im[c]
                val sr = sc.re
                val si = sc.im
                for (y in 0 until height) {
                    val o = y * width + x
                    sr[y] = rc[o]
                    si[y] = ic[o]
                }
                columnFft.transform(sr, si, 0, inverse)
                for (y in 0 until height) {
                    val o = y * width + x
                    rc[o] = sr[y]
                    ic[o] = si[y]
                }
                idx++
            }
        }
    }

    private class FloatPair(val re: FloatArray, val im: FloatArray)
}
