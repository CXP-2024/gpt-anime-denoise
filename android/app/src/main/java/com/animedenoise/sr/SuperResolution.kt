package com.animedenoise.sr

import com.animedenoise.core.RgbaImage
import java.io.Closeable

/** A Real-ESRGAN-style x4 network. */
interface SuperResolution : Closeable {
    /** Native upscale factor of the network. */
    val scale: Int

    /** Which accelerator actually served the graph: "CPU", "NNAPI" or "QNN". */
    val backendName: String

    /** Runs the network on [image] (RGBA, alpha ignored) and returns an image scaled by [scale]. */
    fun upscale(image: RgbaImage): RgbaImage

    /** Free native memory. Safe to call twice. */
    override fun close()
}
