package com.animedenoise.core

/**
 * The whole denoise pipeline as one call, for tests, benchmarks and any caller that does
 * not need per-stage timings or progress reporting.
 *
 * It is the direct equivalent of `denoise.sr_at` plus `l0_smooth`:
 *
 * ```python
 * small = src.resize((round(W * 0.6), round(H * 0.6)), Image.BOX)   # box average
 * big   = upscale(model, small)                                      # Real-ESRGAN x4
 * back  = big.resize((W, H), Image.LANCZOS)                          # back to full size
 * l0    = to_img(l0_smooth(to_arr(back), lam))                       # L0 smoothing
 * ```
 *
 * The super-resolution model is supplied by the caller: this library is pure JVM and the
 * ONNX session only exists in the Android app. [superResolve] receives the downscaled
 * image and must return the enlarged one; its output size is not important, because the
 * next stage resizes to the original size anyway.
 *
 * The Android app does not use this class - it drives the stages itself so it can report
 * progress and reuse an ONNX session between runs - but the stages are the same objects,
 * so behaviour is identical.
 */
object Pipeline {

    /** Downscale factor of the default method, `denoise.DOWN`. */
    const val DEFAULT_DOWN = 0.6

    /** L0 strength of the default method, `denoise.L0`. */
    const val DEFAULT_LAMBDA = 0.001

    /**
     * Runs the full pipeline on an 8-bit image.
     *
     * The 8-bit domain is preserved end to end: the resize stages are byte-exact PIL ports
     * and only the L0 stage needs floats, so the result is what the Python tool would
     * produce, not an approximation of it.
     *
     * `param src          the source image, RGBA bytes as they come out of a bitmap or
     *                     a PNG decoder
     * `param superResolve super-resolution stage: receives the downscaled image and returns
     *                     an enlarged one, usually x4
     * `param down         downscale factor before super-resolution, 0.6 by default
     * `param lam          L0 strength, 0.001 by default
     * `param onStage      called with `downscale`, `superres`, `resize`, `l0` and `encode`
     *                     before each stage, for progress reporting
     * `param isCancelled  polled between stages and once per L0 iteration; when it returns
     *                     `true` the call throws [L0CancelledException] (or returns early if
     *                     the cancellation is seen outside the L0 loop)
     * `return a new image of exactly the source size
     */
    fun denoise(
        src: RgbaImage,
        superResolve: (RgbaImage) -> RgbaImage,
        down: Double = DEFAULT_DOWN,
        lam: Double = DEFAULT_LAMBDA,
        onStage: (String) -> Unit = {},
        isCancelled: () -> Boolean = { false }
    ): RgbaImage {
        require(down > 0.0 && down <= 1.0) { "down must be in (0, 1], was " + down }
        val width = src.width
        val height = src.height

        onStage("downscale")
        val smallW = PilResize.pythonRoundSize(width, down)
        val smallH = PilResize.pythonRoundSize(height, down)
        val small = PilResize.resize(src, smallW, smallH, PilResize.Filter.BOX)
        if (isCancelled()) throw L0CancelledException()

        onStage("superres")
        val big = superResolve(small)
        if (isCancelled()) throw L0CancelledException()

        onStage("resize")
        val back = PilResize.resize(big, width, height, PilResize.Filter.LANCZOS)
        if (isCancelled()) throw L0CancelledException()

        onStage("l0")
        val floats = FloatImage.fromRgba(back)
        val smoothed = L0Smoother.smooth(floats, lam, isCancelled = isCancelled)

        onStage("encode")
        // The reference keeps transparency in a side channel and re-attaches it after
        // processing (open_image / save_png), so the source alpha goes straight through.
        return FloatImage.toRgba(smoothed, src)
    }

    /**
     * The geometry half of the pipeline only: box downscale, super-resolve, LANCZOS back.
     *
     * Useful when the app wants to reuse the intermediate result across several L0
     * strengths, which is exactly what the desktop GUI does when only the slider moves.
     *
     * `param src          source image
     * `param superResolve super-resolution stage
     * `param down         downscale factor, 0.6 by default
     * `return an image of the source size, before L0
     */
    fun preprocess(
        src: RgbaImage,
        superResolve: (RgbaImage) -> RgbaImage,
        down: Double = DEFAULT_DOWN
    ): RgbaImage {
        require(down > 0.0 && down <= 1.0) { "down must be in (0, 1], was " + down }
        val smallW = PilResize.pythonRoundSize(src.width, down)
        val smallH = PilResize.pythonRoundSize(src.height, down)
        val small = PilResize.resize(src, smallW, smallH, PilResize.Filter.BOX)
        return PilResize.resize(superResolve(small), src.width, src.height, PilResize.Filter.LANCZOS)
    }

    /**
     * The smoothing half of the pipeline only: 8-bit in, L0, 8-bit out.
     *
     * This is `to_img(l0_smooth(to_arr(img), lam))`, the code path the reference uses for
     * the `--all` comparison sheet - and the same path the GUI's local-protection brush
     * blends over.
     *
     * `param src 8-bit image, typically the output of [preprocess]
     * `param lam L0 strength
     * `param isCancelled polled once per L0 iteration
     */
    fun smoothOnly(
        src: RgbaImage,
        lam: Double = DEFAULT_LAMBDA,
        isCancelled: () -> Boolean = { false }
    ): RgbaImage {
        val floats = FloatImage.fromRgba(src)
        return FloatImage.toRgba(L0Smoother.smooth(floats, lam, isCancelled = isCancelled), src)
    }
}
