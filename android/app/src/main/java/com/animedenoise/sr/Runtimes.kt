package com.animedenoise.sr

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.util.Log
import java.util.EnumSet

/**
 * Builds ONNX Runtime sessions for each acceleration route and reports which one actually worked.
 *
 * Routes, in the order they are tried:
 *
 *  - **XNNPACK**: CPU kernels tuned for ARM. On a phone this is usually the fastest *reliable* option,
 *    because it keeps the whole graph on one optimized backend instead of splitting it.
 *  - **QNN**: Qualcomm's stack, only in the *-qnn AAR. It has two devices, and they need different
 *    model formats: the Hexagon NPU (htp) wants quantized graphs, while the Adreno GPU (gpu) runs fp32
 *    through OpenCL. This app ships fp32 graphs, so only the GPU device is worth attempting.
 *  - **NNAPI**: deprecated by Google; on real hardware it usually falls back to CPU and is slower than
 *    plain CPU. Kept only so the UI can show what the phone reports.
 *
 * Nothing here throws for a missing accelerator: [create] falls through and records why.
 */
object Runtimes {

    private const val TAG = "Runtimes"

    enum class Backend(val label: String) {
        /** ARM CPU kernels from ONNX Runtime's XNNPACK provider. */
        XNNPACK("XNNPACK"),

        /** Plain CPU kernels, the correctness reference. */
        CPU("CPU"),

        /** Qualcomm Adreno GPU through QNN's OpenCL backend. */
        QNN_GPU("Adreno GPU"),

        /** Qualcomm Hexagon NPU. Needs a quantized model, so it will reject this one. */
        QNN_NPU("Hexagon NPU"),

        /** Deprecated by Google; kept for measurement only. */
        NNAPI("NNAPI"),
        ;

        /** Whether this route is expected to work with the fp32 graphs shipped in assets. */
        val expectsFp32: Boolean get() = this != QNN_NPU
    }

    data class Built(
        val runtime: OnnxSuperResolution,
        val requested: Backend,
        /** Fallback note when [requested] could not be used. */
        val note: String?,
    )

    /**
     * Default order when the caller has no preference.
     *
     * Measured on a Redmi Note 12 Turbo, 307x307 -> 1228x1228 with the 6B graph:
     *
     *   CPU      11.4 s
     *   XNNPACK  13.2 s
     *   NNAPI    13.2 s
     *
     * XNNPACK is compiled into this AAR, but it partitions the graph and its fallback nodes run through
     * the default CPU path anyway, so it ends up slower than letting ORT optimize the whole graph for
     * ARM. CPU therefore stays the default, and XNNPACK remains reachable only for comparison.
     */
    val DEFAULT_ORDER = listOf(Backend.CPU, Backend.XNNPACK)

    /**
     * @param modelBytes the .onnx file contents
     * @param preferred which accelerator to try first; the rest of a sane chain is appended
     */
    fun create(
        env: OrtEnvironment,
        model: SrModel,
        modelBytes: ByteArray,
        preferred: Backend,
        bandRows: Int = model.defaultBandRows,
    ): Built {
        var note: String? = null
        val order = chainFor(preferred)
        for (backend in order) {
            val options = try {
                sessionOptions(backend)
            } catch (t: Throwable) {
                note = (note ?: "") + backend.label + " options failed (" + t.message + "). "
                null
            }
            if (options == null) {
                note = (note ?: "") + backend.label + " is not available in this build. "
                continue
            }
            try {
                val session = env.createSession(modelBytes, options)
                Log.i(TAG, "loaded " + model.assetName + " on " + backend.label)
                return Built(
                    OnnxSuperResolution(env, session, model.scale, backend.label, model.halo, bandRows),
                    backend,
                    note?.trim(),
                )
            } catch (t: Throwable) {
                note = (note ?: "") + backend.label + " failed (" + (t.message ?: t.javaClass.simpleName) + "). "
                Log.w(TAG, "session on " + backend.label + " failed", t)
            } finally {
                // SessionOptions only configures creation; ORT copies what it needs, so release it.
                options.close()
            }
        }
        error("no ONNX Runtime backend could load " + model.assetName + (note?.let { ": " + it } ?: ""))
    }

    /**
     * The fallback chain for a requested backend.
     *
     * QNN is deliberately absent: on the test device both its devices crash or reject the fp32 graph, and
     * a native crash inside createSession takes the whole process down with it. QNN is only reachable
     * through [QnnProbeService], which runs in a separate process precisely so it cannot do that.
     */
    private fun chainFor(preferred: Backend): List<Backend> = when (preferred) {
        Backend.QNN_NPU, Backend.QNN_GPU -> listOf(Backend.QNN_GPU, Backend.QNN_NPU, Backend.XNNPACK, Backend.CPU)
        Backend.XNNPACK -> listOf(Backend.XNNPACK, Backend.CPU)
        Backend.CPU -> listOf(Backend.CPU)
        Backend.NNAPI -> listOf(Backend.NNAPI, Backend.XNNPACK, Backend.CPU)
    }

    /** Null when the accelerator is not compiled into this ORT build. */
    private fun sessionOptions(backend: Backend): OrtSession.SessionOptions? {
        val options = OrtSession.SessionOptions()
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        val cores = Runtime.getRuntime().availableProcessors()
        options.setInterOpNumThreads(1)
        options.setIntraOpNumThreads(maxOf(2, cores - 1))
        when (backend) {
            Backend.CPU -> return options

            Backend.XNNPACK -> {
                // Empty map = provider defaults (multi-threaded ARM kernels).
                return try {
                    options.addXnnpack(emptyMap())
                    options
                } catch (t: Throwable) {
                    Log.i(TAG, "XNNPACK not compiled in: " + t.message)
                    options.close()
                    null
                }
            }

            Backend.NNAPI -> {
                return try {
                    options.addNnapi(EnumSet.of(NNAPIFlags.CPU_DISABLED).takeIf { false } ?: EnumSet.noneOf(NNAPIFlags::class.java))
                    options
                } catch (t: Throwable) {
                    Log.i(TAG, "NNAPI is not compiled into this ORT build: " + t.message)
                    options.close()
                    null
                }
            }

            // QNN ships only with the *-qnn AAR, which is not part of this build: it segfaulted
            // natively inside createSession on the test device and tripled the APK size.
            Backend.QNN_GPU, Backend.QNN_NPU -> {
                Log.i(TAG, "QNN is not part of this build")
                options.close()
                return null
            }
        }
    }
}
