package com.animedenoise.ui

import android.app.Application
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.animedenoise.DenoiseEngine
import com.animedenoise.core.RgbaImage
import com.animedenoise.image.ImageIo
import com.animedenoise.DenoiseEngine.Companion.PRESETS
import com.animedenoise.image.Persistence
import com.animedenoise.sr.ModelStore
import com.animedenoise.sr.Runtimes
import com.animedenoise.sr.SrModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Everything the single screen renders. */
data class UiState(
    val source: RgbaImage? = null,
    val sourceBitmap: Bitmap? = null,
    val sourceName: String? = null,
    val sourceUri: Uri? = null,
    val result: RgbaImage? = null,
    val resultBitmap: Bitmap? = null,
    val busy: Boolean = false,
    val stage: String = "",
    val error: String? = null,
    val presetIndex: Int = 1,
    /** Overrides for the fine-tune row; null means "use the preset value". */
    val down: Double? = null,
    val lam: Double? = null,
    val compareFraction: Float = 0.5f,
    val viewMode: ViewMode = ViewMode.COMPARE,
    val backend: String = "",
    val lastMillis: Long = 0,
    val stageMillis: Map<String, Long> = emptyMap(),
    val savedUri: Uri? = null,
    val info: String = "",
    val models: List<SrModel> = emptyList(),
    val modelIndex: Int = 0,
    val accelerate: Boolean = true,
    /** True when the last run skipped the model pass because only L0 changed. */
    val reusedSuperResolution: Boolean = false,
) {
    enum class ViewMode { COMPARE, ORIGINAL, RESULT }

    val preset: DenoiseEngine.Companion.Preset get() = PRESETS[presetIndex]
    val effectiveDown: Double get() = down ?: preset.down
    val effectiveLam: Double get() = lam ?: preset.lam
    val model: SrModel? get() = models.getOrNull(modelIndex)
}

/**
 * Owns the heavy pipeline.
 *
 * Only one job runs at a time: changing a preset or picking a new picture cancels the running job and
 * restarts, which is what the desktop web app does too.
 */
class DenoiseViewModel(app: Application) : AndroidViewModel(app) {

    private val engine = DenoiseEngine(app)
    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var job: Job? = null
    @Volatile private var cancelled = false

    init {
        val bundled = ModelStore.available(getApplication<Application>().assets)
        if (bundled.isNotEmpty()) {
            _state.update { it.copy(models = SrModel.bundled(bundled), modelIndex = 0) }
        }
    }

    /**
     * Loads an already-decoded image and starts processing.
     *
     * Used by the self test to drive the visible UI through the same flow a gallery pick would,
     * so a long run shows real progress instead of looking frozen.
     */
    fun loadDirect(image: com.animedenoise.core.RgbaImage, name: String?) {
        android.util.Log.i("DenoiseVM", "loadDirect called: " + image.width + "x" + image.height)
        cancel()
        _state.update { it.copy(busy = true, stage = "正在读入原图", error = null, savedUri = null) }
        viewModelScope.launch {
            try {
                val bitmap = withContext(Dispatchers.Default) { ImageIo.toBitmap(image) }
                _state.update {
                    it.copy(
                        source = image,
                        sourceBitmap = bitmap,
                        sourceName = name,
                        sourceUri = null,
                        result = null,
                        resultBitmap = null,
                        busy = false,
                        stage = "",
                        info = image.width.toString() + "×" + image.height,
                    )
                }
                android.util.Log.i("DenoiseVM", "loadDirect state set, calling process")
                process()
            } catch (t: Throwable) {
                android.util.Log.w("DenoiseVM", "loadDirect failed", t)
                fail(t)
            }
        }
    }

    fun pick(uri: Uri, name: String?) {
        cancel()
        _state.update { it.copy(busy = true, stage = "正在读入原图", error = null, savedUri = null) }
        viewModelScope.launch {
            try {
                val image = withContext(Dispatchers.IO) {
                    val resolver = getApplication<Application>().contentResolver
                    ImageIo.decodeRgba(resolver, uri)
                }
                val bitmap = withContext(Dispatchers.Default) { ImageIo.toBitmap(image) }
                _state.update {
                    it.copy(
                        source = image,
                        sourceBitmap = bitmap,
                        sourceName = name ?: uri.lastPathSegment,
                        sourceUri = uri,
                        result = null,
                        resultBitmap = null,
                        busy = false,
                        stage = "",
                        info = image.width.toString() + "×" + image.height,
                    )
                }
                process()
            } catch (t: Throwable) {
                fail(t)
            }
        }
    }

    fun selectPreset(index: Int) {
        if (index == _state.value.presetIndex) return
        _state.update { it.copy(presetIndex = index, down = null, lam = null) }
        if (_state.value.source != null) process()
    }

    /** Drag in progress: update the readout only, so a slider sweep is not dozens of model runs. */
    fun previewDown(value: Double) = _state.update { it.copy(down = value) }

    fun previewLam(value: Double) = _state.update { it.copy(lam = value) }

    /** Slider released: apply and recompute. */
    fun commitDown(value: Double) {
        _state.update { it.copy(down = value) }
        if (_state.value.source != null) process()
    }

    fun commitLam(value: Double) {
        _state.update { it.copy(lam = value) }
        if (_state.value.source != null) process()
    }

    fun selectModel(index: Int) {
        if (index == _state.value.modelIndex) return
        _state.update { it.copy(modelIndex = index) }
        if (_state.value.source != null) process()
    }

    fun setAccelerate(enabled: Boolean) {
        if (enabled == _state.value.accelerate) return
        _state.update { it.copy(accelerate = enabled) }
        if (_state.value.source != null) process()
    }

    fun setViewMode(mode: UiState.ViewMode) = _state.update { it.copy(viewMode = mode) }

    fun setCompareFraction(fraction: Float) =
        _state.update { it.copy(compareFraction = fraction.coerceIn(0f, 1f)) }

    fun clearSavedFlag() = _state.update { it.copy(savedUri = null, error = null) }

    /** Re-runs the current picture with the current settings. */
    fun process() {
        val current = _state.value
        val source = current.source ?: return
        val model = current.model ?: return
        cancel()
        cancelled = false
        _state.update { it.copy(busy = true, error = null, savedUri = null) }
        job = viewModelScope.launch {
            try {
                // The toggle chooses between the plain CPU kernels and XNNPACK. Earlier AARs
                // lacked XNNPACK entirely; the current one has it, and measurements on the test
                // device showed XNNPACK within a few percent of plain CPU on this graph, so the
                // switch is honest again: both routes work, they just differ in kernel selection.
                val backend = if (current.accelerate) Runtimes.Backend.XNNPACK else Runtimes.Backend.CPU
                val name = withContext(Dispatchers.Default) {
                    engine.ensureModel(model, backend)
                }
                _state.update { it.copy(backend = name) }
                val down = current.effectiveDown
                val lam = current.effectiveLam
                val sourceUsesAlpha = withContext(Dispatchers.Default) {
                    ImageIo.usesAlpha(source)
                }
                val result = withContext(Dispatchers.Default) {
                    engine.denoise(
                        src = source,
                        down = down,
                        lam = lam,
                        sourceHasAlpha = sourceUsesAlpha,
                        onStage = { text -> _state.update { it.copy(stage = text) } },
                        isCancelled = { cancelled },
                    )
                }
                val bitmap = withContext(Dispatchers.Default) { ImageIo.toBitmap(result.output) }
                val total = result.stageMillis.values.sum()
                _state.update {
                    it.copy(
                        result = result.output,
                        resultBitmap = bitmap,
                        busy = false,
                        stage = "",
                        backend = result.backend,
                        lastMillis = total,
                        stageMillis = result.stageMillis,
                        reusedSuperResolution = result.reusedSuperResolution,
                        info = it.source?.let { s -> s.width.toString() + "×" + s.height } ?: "",
                    )
                }
            } catch (c: DenoiseEngine.Cancelled) {
                Log.i(TAG, "cancelled")
            } catch (t: Throwable) {
                fail(t)
            }
        }
    }

    fun saveNew() {
        val current = _state.value
        val output = current.result ?: return
        viewModelScope.launch {
            val uri = withContext(Dispatchers.IO) {
                val png = ImageIo.encodePng(output)
                val name = Persistence.suggestName(current.sourceName, methodTag(current))
                Persistence.saveToPictures(
                    getApplication<Application>().contentResolver, name, png,
                )
            }
            _state.update {
                it.copy(
                    savedUri = uri,
                    error = if (uri == null) "保存失败，请检查存储空间" else null,
                )
            }
        }
    }

    fun overwriteOriginal() {
        val current = _state.value
        val output = current.result ?: return
        val uri = current.sourceUri ?: return
        viewModelScope.launch {
            val ok = withContext(Dispatchers.IO) {
                Persistence.overwrite(
                    getApplication<Application>().contentResolver, uri, ImageIo.encodePng(output),
                )
            }
            _state.update {
                it.copy(
                    savedUri = if (ok) uri else null,
                    error = if (ok) null else "原文件不允许写入，请用「另存为」",
                )
            }
        }
    }

    fun shareIntent(): Intent? {
        val current = _state.value
        val output = current.result ?: return null
        val png = ImageIo.encodePng(output)
        val file = Persistence.saveLocal(
            java.io.File(getApplication<Application>().cacheDir, "share"),
            Persistence.suggestName(current.sourceName, methodTag(current)),
            png,
        )
        val app = getApplication<Application>()
        val uri = androidx.core.content.FileProvider.getUriForFile(
            app, app.packageName + ".files", file,
        )
        return Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }

    fun cancel() {
        cancelled = true
        job?.cancel()
        job = null
        if (_state.value.busy) _state.update { it.copy(busy = false, stage = "") }
    }

    private fun fail(t: Throwable) {
        Log.w(TAG, "pipeline failed: uri=" + _state.value.sourceUri + " name=" + _state.value.sourceName, t)
        _state.update {
            it.copy(busy = false, stage = "", error = t.message ?: t.javaClass.simpleName)
        }
    }

    private fun methodTag(state: UiState): String {
        val down = (state.effectiveDown * 100).toInt()
        return "esrgan" + down + "_l0_" + trimLam(state.effectiveLam)
    }

    private fun trimLam(lam: Double): String {
        val text = String.format(java.util.Locale.US, "%.4f", lam)
        return text.trimEnd('0').trimEnd('.')
    }

    override fun onCleared() {
        cancel()
        engine.release()
        super.onCleared()
    }

    companion object {
        private const val TAG = "DenoiseVM"
    }
}
