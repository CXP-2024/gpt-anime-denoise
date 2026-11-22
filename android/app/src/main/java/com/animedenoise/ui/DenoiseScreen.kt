package com.animedenoise.ui

import android.net.Uri
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.animedenoise.DenoiseEngine
import com.animedenoise.DenoiseEngine.Companion.PRESETS
import java.util.Locale

/**
 * Single-screen flow: the image fills the screen, controls dock at the bottom.
 *
 * While a run is in progress the dock is replaced by a step list (读图 → 重建细节 → 压平噪点 →
 * 完成) so the user always knows which of the four pipeline stages is running and roughly how far
 * along it is. The stage text comes straight from the engine, which knows the strip index.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DenoiseScreen(
    viewModel: DenoiseViewModel,
    onPickImage: (onPicked: (Uri?) -> Unit) -> Unit,
) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val snackbar = remember { SnackbarHostState() }
    val hasImage = state.source != null

    LaunchedEffect(state.error, state.savedUri) {
        state.error?.let { snackbar.showSnackbar(it); viewModel.clearSavedFlag() }
        if (state.savedUri != null && state.error == null) {
            snackbar.showSnackbar("已保存到相册 Pictures/AnimeDenoise")
            viewModel.clearSavedFlag()
        }
    }

    Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        CompareView(
            original = state.sourceBitmap,
            result = state.resultBitmap,
            mode = state.viewMode,
            fraction = state.compareFraction,
            onFractionChange = viewModel::setCompareFraction,
            modifier = Modifier.fillMaxSize(),
        )

        Row(
            Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "二次元去噪工具",
                fontWeight = FontWeight.SemiBold,
                color = Color.White,
                modifier = Modifier
                    .padding(horizontal = 12.dp, vertical = 8.dp)
                    .clip(RoundedCornerShape(50))
                    .background(Color.Black.copy(alpha = 0.35f))
                    .padding(horizontal = 12.dp, vertical = 6.dp),
            )
            Spacer(Modifier.weight(1f))
            if (hasImage) {
                FloatingIcon(
                    icon = Icons.Default.Refresh,
                    desc = "重新处理",
                    enabled = !state.busy,
                    onClick = { viewModel.process() },
                )
                FloatingIcon(
                    icon = Icons.Default.Share,
                    desc = "分享",
                    enabled = !state.busy && state.result != null,
                    onClick = {
                        viewModel.shareIntent()?.let { intent ->
                            ContextCompat.startActivity(
                                context,
                                android.content.Intent.createChooser(intent, "分享结果"),
                                null,
                            )
                        }
                    },
                )
            }
            FloatingIcon(
                icon = Icons.Default.AddPhotoAlternate,
                desc = "选择图片",
                enabled = true,
                onClick = { onPickImage { uri -> uri?.let { viewModel.pick(it, null) } } },
            )
        }

        // Progress overlay with the step list. Visible only while busy.
        AnimatedVisibility(
            visible = state.busy,
            enter = fadeIn() + slideInVertically { it / 4 },
            exit = fadeOut() + slideOutVertically { it / 4 },
            modifier = Modifier.align(Alignment.Center),
        ) {
            Surface(
                shape = RoundedCornerShape(24.dp),
                color = Color.Black.copy(alpha = 0.72f),
            ) {
                Column(
                    Modifier.padding(horizontal = 26.dp, vertical = 22.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    LinearProgressIndicator(
                        modifier = Modifier.size(width = 220.dp, height = 5.dp),
                        color = MaterialTheme.colorScheme.primary,
                        trackColor = Color.White.copy(alpha = 0.18f),
                    )
                    Spacer(Modifier.height(18.dp))
                    StepList(state.stage)
                }
            }
        }

        Surface(
            modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth(),
            shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp),
            color = MaterialTheme.colorScheme.surface,
            tonalElevation = 3.dp,
        ) {
            Column(
                Modifier.navigationBarsPadding().padding(horizontal = 20.dp, vertical = 14.dp),
            ) {
                if (!hasImage) {
                    EmptyDock { onPickImage { uri -> uri?.let { viewModel.pick(it, null) } } }
                } else {
                    LoadedDock(state, viewModel)
                }
            }
        }

        SnackbarHost(
            snackbar,
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 8.dp),
        )
    }
}

/**
 * The four pipeline stages rendered as a checklist.
 *
 * [stageText] is the engine's free-form progress string; matching against its keywords keeps the
 * engine free of UI concerns while still driving which row is highlighted.
 */
@Composable
private fun StepList(stageText: String) {
    val steps = listOf(
        "读取图片" to "读入原图",
        "重建" to "AI 重建画面细节",
        "压平" to "压平残留噪点",
        "整理" to "输出成品",
    )
    val activeIndex = when {
        stageText.contains("重建") -> 1
        stageText.contains("压平") -> 2
        stageText.contains("整理") -> 3
        else -> 0
    }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        steps.forEachIndexed { index, (keyword, label) ->
            val state = when {
                index < activeIndex -> StepState.DONE
                index == activeIndex -> StepState.ACTIVE
                else -> StepState.PENDING
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                val tint = when (state) {
                    StepState.DONE -> MaterialTheme.colorScheme.primary
                    StepState.ACTIVE -> MaterialTheme.colorScheme.primary
                    StepState.PENDING -> Color.White.copy(alpha = 0.35f)
                }
                if (state == StepState.DONE) {
                    Icon(
                        Icons.Default.Check,
                        contentDescription = null,
                        tint = tint,
                        modifier = Modifier.size(18.dp),
                    )
                } else {
                    Box(
                        Modifier
                            .size(18.dp)
                            .clip(RoundedCornerShape(50))
                            .background(
                                if (state == StepState.ACTIVE) MaterialTheme.colorScheme.primary
                                else Color.White.copy(alpha = 0.25f)
                            )
                    )
                }
                Spacer(Modifier.size(10.dp))
                Text(
                    label,
                    color = if (state == StepState.PENDING) Color.White.copy(alpha = 0.55f) else Color.White,
                    fontWeight = if (state == StepState.ACTIVE) FontWeight.SemiBold else FontWeight.Normal,
                    style = MaterialTheme.typography.bodyMedium,
                )
                if (state == StepState.ACTIVE) {
                    Spacer(Modifier.size(8.dp))
                    Text(
                        stageText,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
            }
        }
    }
}

private enum class StepState { DONE, ACTIVE, PENDING }

/** Circular translucent icon button used over the image. */
@Composable
private fun FloatingIcon(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    desc: String,
    enabled: Boolean,
    onClick: () -> Unit,
) {
    IconButton(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier
            .padding(start = 6.dp)
            .size(42.dp)
            .clip(RoundedCornerShape(50))
            .background(Color.Black.copy(alpha = 0.35f)),
        colors = IconButtonDefaults.iconButtonColors(
            contentColor = Color.White,
            disabledContentColor = Color.White.copy(alpha = 0.35f),
        ),
    ) {
        Icon(icon, contentDescription = desc, modifier = Modifier.size(22.dp))
    }
}

/** The dock before any image is loaded: one clear call to action. */
@Composable
private fun EmptyDock(onPick: () -> Unit) {
    Column(
        Modifier.fillMaxWidth().padding(vertical = 10.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            "去掉 GPT 生图在二次元画面里的噪点",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(4.dp))
        Text(
            "图片只在本机处理，不会上传",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(14.dp))
        Button(
            onClick = onPick,
            modifier = Modifier.fillMaxWidth().height(52.dp),
            shape = RoundedCornerShape(14.dp),
        ) {
            Icon(Icons.Default.AddPhotoAlternate, contentDescription = null)
            Spacer(Modifier.size(8.dp))
            Text("选择图片")
        }
    }
}

/** The dock once an image is loaded: view switcher, presets, sliders. */
@Composable
private fun LoadedDock(state: UiState, viewModel: DenoiseViewModel) {
    Column(
        Modifier
            .fillMaxWidth()
            .heightIn(max = 320.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            SingleChoiceSegmentedButtonRow(Modifier.weight(1f)) {
                UiState.ViewMode.entries.forEachIndexed { index, mode ->
                    SegmentedButton(
                        selected = state.viewMode == mode,
                        onClick = { viewModel.setViewMode(mode) },
                        shape = SegmentedButtonDefaults.itemShape(index, UiState.ViewMode.entries.size),
                        enabled = state.result != null,
                        label = {
                            Text(
                                when (mode) {
                                    UiState.ViewMode.COMPARE -> "对比"
                                    UiState.ViewMode.ORIGINAL -> "原图"
                                    UiState.ViewMode.RESULT -> "结果"
                                }
                            )
                        },
                    )
                }
            }
            IconButton(
                onClick = { viewModel.saveNew() },
                enabled = state.result != null,
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = MaterialTheme.colorScheme.onPrimary,
                ),
            ) {
                Icon(Icons.Default.Save, contentDescription = "另存为")
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            PRESETS.forEachIndexed { index, preset ->
                FilterChip(
                    selected = state.presetIndex == index,
                    onClick = { viewModel.selectPreset(index) },
                    label = {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(preset.name, fontWeight = if (state.presetIndex == index) FontWeight.SemiBold else FontWeight.Normal)
                            Text(
                                (preset.down * 100).toInt().toString() + "% · " + trimLam(preset.lam),
                                style = MaterialTheme.typography.labelSmall,
                            )
                        }
                    },
                )
            }
        }
        AnimatedHint(text = state.preset.hint)

        FineTune(
            label = "预缩比例",
            value = state.effectiveDown,
            range = 0.3f..0.95f,
            format = { (it * 100).toInt().toString() + "%" },
            onPreview = viewModel::previewDown,
            onCommit = viewModel::commitDown,
        )
        FineTune(
            label = "L0 平滑 · 改这项会复用重建结果",
            value = state.effectiveLam,
            range = 0.0002f..0.004f,
            format = { String.format(Locale.US, "%.4f", it) },
            onPreview = viewModel::previewLam,
            onCommit = viewModel::commitLam,
        )

        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text("硬件加速", style = MaterialTheme.typography.bodyMedium)
                Text(
                    state.backend.ifBlank { "CPU" } +
                        if (state.lastMillis > 0)
                            " · " + String.format(Locale.US, "%.1f", state.lastMillis / 1000.0) + " 秒" +
                            (if (state.reusedSuperResolution) "（复用重建）" else "")
                        else "",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Switch(checked = state.accelerate, onCheckedChange = viewModel::setAccelerate)
        }
    }
}

/** Small text that fades when its content changes, so hints do not pop. */
@Composable
private fun AnimatedHint(text: String) {
    AnimatedContent(
        targetState = text,
        transitionSpec = { fadeIn().togetherWith(fadeOut()) },
        label = "hint",
    ) { value ->
        Text(
            value,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/**
 * A labelled slider whose heavy work happens on release.
 *
 * [onPreview] updates the numeric readout while dragging; [onCommit] runs once on release, so a
 * sweep does not queue dozens of cancelled jobs.
 */
@Composable
private fun FineTune(
    label: String,
    value: Double,
    range: ClosedFloatingPointRange<Float>,
    format: (Float) -> String,
    onPreview: (Double) -> Unit,
    onCommit: (Double) -> Unit,
) {
    var dragging by remember { mutableStateOf(false) }
    var local by remember(value) { mutableFloatStateOf(value.toFloat()) }
    Column {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, style = MaterialTheme.typography.bodySmall)
            Text(
                format(if (dragging) local else value.toFloat()),
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.primary,
            )
        }
        Slider(
            value = if (dragging) local else value.toFloat().coerceIn(range.start, range.endInclusive),
            onValueChange = {
                dragging = true
                local = it
                onPreview(it.toDouble())
            },
            onValueChangeFinished = {
                dragging = false
                onCommit(local.toDouble())
            },
            valueRange = range,
        )
    }
}

private fun trimLam(lam: Double): String {
    val text = String.format(Locale.US, "%.4f", lam)
    return text.trimEnd('0').trimEnd('.')
}
