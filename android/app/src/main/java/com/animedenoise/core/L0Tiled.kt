package com.animedenoise.core

/**
 * Large-image L0 through overlapping tiles.
 *
 * The L0 solve is a global frequency-domain optimisation, so it cannot be split naively. What works
 * is overlapping tiles with a generous halo: the solve's influence decays quickly outside its own
 * tile, so a tile computed on an expanded window and cropped to its centre is very close to the
 * whole-image result. Measured on the real 2160x3840 image at 1024x1024 test size, halo 128:
 * 1.6% of pixels differ by more than one output level and 0.19% by more than two - an order of
 * magnitude below the v3-vs-6B model gap the app already ships.
 *
 * Why tiles at all: the whole-image solve needs ~1 GB of working planes at 8.3 MP, which on a phone
 * means swap thrashing and a lowmemorykiller strike. A single tile of [TILE] + 2*[HALO] is tens of
 * megabytes, so it lives comfortably in the Java heap and never touches swap.
 */
object L0Tiled {

    /**
     * Central region processed per tile.
     *
     * Overhead per tile is (TILE + 2*HALO)^2 / TILE^2, so this is the main speed knob: 768 wastes
     * 78% of each solve on halo, 1024 wastes 56%, 1280 wastes 44%. Larger tiles also mean a larger
     * working set, but a 1152x1152 float solve is still only a few tens of MB.
     */
    const val TILE = 1024

    /** Context rows/columns around each tile. Measured: 128 gives 1.6% of pixels >1 level. */
    const val HALO = 128

    /** Tiles are processed one at a time; the smoother uses the multi-core pool inside. */
    fun smooth(
        img: FloatImage,
        lam: Double,
        kappa: Double = 2.0,
        betaMax: Double = L0SmootherF32.DEFAULT_BETA_MAX,
        isCancelled: () -> Boolean = { false },
    ): FloatImage {
        val w = img.width
        val h = img.height

        // Small enough for one pass: no tiling overhead, use the plain path.
        if (w <= TILE && h <= TILE) {
            return L0SmootherF32.smooth(img, lam, kappa, betaMax, isCancelled)
        }

        val out = FloatImage(w, h)
        val outR = out.r
        val outG = out.g
        val outB = out.b

        var y0 = 0
        while (y0 < h) {
            val y1 = minOf(h, y0 + TILE)
            var x0 = 0
            while (x0 < w) {
                val x1 = minOf(w, x0 + TILE)
                if (isCancelled()) throw L0CancelledException()

                val ay0 = maxOf(0, y0 - HALO)
                val ax0 = maxOf(0, x0 - HALO)
                val ay1 = minOf(h, y1 + HALO)
                val ax1 = minOf(w, x1 + HALO)
                val tw = ax1 - ax0
                val th = ay1 - ay0

                // Copy the expanded window into a small FloatImage.
                val tile = FloatImage(tw, th)
                copyRegion(img, tile, ax0, ay0, tw, th)

                val solved = L0SmootherF32.smooth(tile, lam, kappa, betaMax, isCancelled)

                // Paste back only the tile's own centre region.
                pasteRegion(solved, outR, outG, outB, w,
                    x0, y0, x1, y1, ax0, ay0)
                x0 = x1
            }
            y0 = y1
        }
        return out
    }

    private fun copyRegion(src: FloatImage, dst: FloatImage, ax0: Int, ay0: Int, tw: Int, th: Int) {
        val sw = src.width
        val sr = src.r
        val sg = src.g
        val sb = src.b
        val dr = dst.r
        val dg = dst.g
        val db = dst.b
        for (y in 0 until th) {
            var s = (ay0 + y) * sw + ax0
            var d = y * tw
            for (x in 0 until tw) {
                dr[d] = sr[s]
                dg[d] = sg[s]
                db[d] = sb[s]
                s++
                d++
            }
        }
    }

    private fun pasteRegion(
        src: FloatImage,
        outR: FloatArray,
        outG: FloatArray,
        outB: FloatArray,
        outWidth: Int,
        x0: Int, y0: Int, x1: Int, y1: Int,
        ax0: Int, ay0: Int,
    ) {
        val tw = src.width
        val sr = src.r
        val sg = src.g
        val sb = src.b
        for (y in y0 until y1) {
            var s = (y - ay0) * tw + (x0 - ax0)
            var d = y * outWidth + x0
            for (x in x0 until x1) {
                outR[d] = sr[s]
                outG[d] = sg[s]
                outB[d] = sb[s]
                s++
                d++
            }
        }
    }
}
