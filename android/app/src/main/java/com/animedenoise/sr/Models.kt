package com.animedenoise.sr

/**
 * The neural network shipped in assets/, with the geometry constants derived from its architecture.
 *
 * [halo] is not a tuning knob: it is the receptive radius of the trunk in input pixels. A banded pass
 * reading fewer neighbour rows than the network's own receptive field produces seams, so the value
 * comes from counting the convolutions in the exported graph.
 *
 * Measured on a Redmi Note 12 Turbo (SM7475 / Adreno 725), ONNX Runtime CPU:
 * 2160x3840 source, 60% box pre-shrink, x4 model pass, LANCZOS back.
 */
enum class SrModel(
    /** File name inside assets/. */
    val assetName: String,
    /** Native network factor. */
    val scale: Int,
    /** Input rows of real neighbour context each band needs. */
    val halo: Int,
    /** Rows per inference band when banding is needed. */
    val defaultBandRows: Int,
    /** Shown in the UI. */
    val label: String,
    /** What the user gives up, stated plainly. */
    val hint: String,
    /** Whether this reproduces the desktop tool's reference output. */
    val isReference: Boolean,
) {
    /**
     * Real-ESRGAN animevideov3 (SRVGGNetCompact): about 9x less compute than the 6B ESRGAN, which is
     * what makes a 2160x3840 working image tractable on a phone. Measured against the 6B reference on
     * real anime images: ~90% of pixels differ by 1-2 levels (mean 3.5), concentrated on line work -
     * slightly softer edges, same colour blocks, same eyes.
     */
    ANIMEVIDEOV3(
        assetName = "realesr_animevideov3_x4.onnx",
        scale = 4,
        halo = 24,
        defaultBandRows = 512,
        label = "手机版模型",
        hint = "为手机的速度取舍选择；线条略软于桌面版，色彩与构图一致",
        isReference = false,
    ),
    ;

    companion object {
        /** Models actually bundled in this APK. */
        fun bundled(names: Set<String>): List<SrModel> = entries.filter { it.assetName in names }
    }
}
