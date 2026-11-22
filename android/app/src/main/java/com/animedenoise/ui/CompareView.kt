package com.animedenoise.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.util.Locale

/**
 * Zoomable original-vs-result viewer.
 *
 * Gestures:
 *  - pinch to zoom (up to 16x), drag to pan once zoomed
 *  - COMPARE: drag **anywhere** horizontally to move the split line - a grip sized for a fingertip
 *    is too small a target, and the whole-image drag is what every photo editor does
 *  - press and hold to peek at the original, like holding Z on the desktop viewer
 *
 * The split line itself is drawn by the canvas at [fraction] of the image width; the grip knob is
 * only an affordance and follows the line.
 */
@Composable
fun CompareView(
    original: Bitmap?,
    result: Bitmap?,
    mode: UiState.ViewMode,
    fraction: Float,
    onFractionChange: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    if (original == null) {
        EmptyHint(modifier)
        return
    }
    var zoom by remember { mutableFloatStateOf(1f) }
    var pan by remember { mutableStateOf(Offset.Zero) }
    var boxWidth by remember { mutableFloatStateOf(0f) }
    var peeking by remember { mutableStateOf(false) }

    val originalImage = remember(original) { original.asImageBitmap() }
    val resultImage = remember(result) { result?.asImageBitmap() }

    BoxWithConstraints(
        modifier
            .fillMaxSize()
            .onSizeChanged { boxWidth = it.width.toFloat() },
    ) {
        Box(
            Modifier
                .fillMaxSize()
                .pointerInput(Unit) {
                    detectTransformGestures { _, panChange, zoomChange, _ ->
                        val next = (zoom * zoomChange).coerceIn(1f, 16f)
                        pan = if (next <= 1.001f) Offset.Zero else pan + panChange
                        zoom = next
                    }
                }
                .pointerInput(Unit) {
                    detectTapGestures(
                        onPress = {
                            if (result != null) peeking = true
                            tryAwaitRelease()
                            peeking = false
                        },
                    )
                }
                // Split-line drag lives on the same box as everything else. When in COMPARE mode
                // with a zoom of 1, horizontal drags move the line instead of panning.
                .pointerInput(mode, zoom) {
                    if (mode != UiState.ViewMode.COMPARE || zoom > 1.01f) return@pointerInput
                    detectDragGestures(
                        onDragStart = { start ->
                            if (boxWidth > 0f) onFractionChange((start.x / boxWidth).coerceIn(0f, 1f))
                        },
                    ) { change, _ ->
                        if (boxWidth > 0f) {
                            change.consume()
                            onFractionChange((change.position.x / boxWidth).coerceIn(0f, 1f))
                        }
                    }
                },
        ) {
            Canvas(Modifier.fillMaxSize()) {
                val dst = fitRect(originalImage.width.toFloat(), originalImage.height.toFloat(), size)
                withTransform({
                    translate(dst.left + pan.x, dst.top + pan.y)
                    scale(zoom, zoom, pivot = Offset.Zero)
                }) {
                    val drawSize = Size(dst.width, dst.height)
                    drawCheckerboard(drawSize)
                    val effective = if (peeking) UiState.ViewMode.ORIGINAL else mode
                    when (effective) {
                        UiState.ViewMode.ORIGINAL ->
                            drawImage(originalImage, dstSize = IntSize(drawSize.width.toInt(), drawSize.height.toInt()))

                        UiState.ViewMode.RESULT ->
                            drawImage(resultImage ?: originalImage, dstSize = IntSize(drawSize.width.toInt(), drawSize.height.toInt()))

                        UiState.ViewMode.COMPARE -> {
                            val split = (dst.width * fraction).coerceIn(0f, dst.width)
                            clipRect(right = split) {
                                drawImage(resultImage ?: originalImage, dstSize = IntSize(drawSize.width.toInt(), drawSize.height.toInt()))
                            }
                            clipRect(left = split) {
                                drawImage(originalImage, dstSize = IntSize(drawSize.width.toInt(), drawSize.height.toInt()))
                            }
                            if (split > 0f) {
                                drawLine(
                                    color = Color.White.copy(alpha = 0.92f),
                                    start = Offset(split, 0f),
                                    end = Offset(split, dst.height),
                                    strokeWidth = 1.5.dp.toPx(),
                                )
                            }
                        }
                    }
                }
            }

            // The grip knob follows the line; dragging anywhere already moves it, so this is
            // purely visual.
            if (mode == UiState.ViewMode.COMPARE && !peeking && boxWidth > 0f) {
                SplitGrip(
                    fraction = fraction,
                    boxWidth = boxWidth,
                    modifier = Modifier.align(Alignment.CenterStart),
                )
            }

            if (zoom > 1.01f) {
                Text(
                    text = String.format(Locale.US, "%.1fx", zoom),
                    color = Color.White,
                    fontSize = 11.sp,
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(10.dp)
                        .clip(RoundedCornerShape(50))
                        .background(Color.Black.copy(alpha = 0.45f))
                        .padding(horizontal = 8.dp, vertical = 3.dp),
                )
            }

            Text(
                text = if (peeking) "松手看结果" else labelsFor(mode),
                color = Color.White,
                fontSize = 11.sp,
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(10.dp)
                    .clip(RoundedCornerShape(50))
                    .background(Color.Black.copy(alpha = 0.35f))
                    .padding(horizontal = 8.dp, vertical = 3.dp),
            )
        }
    }
}

private fun labelsFor(mode: UiState.ViewMode): String = when (mode) {
    UiState.ViewMode.COMPARE -> "左右拖动分割线 · 双指缩放"
    UiState.ViewMode.ORIGINAL -> "原图"
    UiState.ViewMode.RESULT -> "结果"
}

/** Visual knob centred on the split line. Dragging anywhere moves the line. */
@Composable
private fun SplitGrip(fraction: Float, boxWidth: Float, modifier: Modifier = Modifier) {
    val density = androidx.compose.ui.platform.LocalDensity.current
    val xDp = with(density) { (fraction * boxWidth).toDp() }
    Box(
        modifier
            .padding(start = xDp - 18.dp)
            .fillMaxSize(),
        contentAlignment = Alignment.Center,
    ) {
        Box(
            Modifier
                .size(36.dp)
                .clip(RoundedCornerShape(50))
                .background(Color.White.copy(alpha = 0.85f)),
            contentAlignment = Alignment.Center,
        ) {
            Text("⇔", color = Color(0xFF1A1A1A), fontSize = 15.sp)
        }
    }
}

@Composable
private fun EmptyHint(modifier: Modifier) {
    Box(modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        Column(
            modifier = Modifier.align(Alignment.Center),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Icon(
                Icons.Default.AddPhotoAlternate,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.outline,
            )
            Spacer(Modifier.height(12.dp))
            Text(
                text = "选择一张图片开始去噪",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/** Letterboxed destination rectangle inside [into], preserving aspect ratio. */
private fun fitRect(w: Float, h: Float, into: Size): androidx.compose.ui.geometry.Rect {
    if (w <= 0f || h <= 0f) {
        return androidx.compose.ui.geometry.Rect(0f, 0f, into.width, into.height)
    }
    val scale = minOf(into.width / w, into.height / h)
    val dw = w * scale
    val dh = h * scale
    val left = (into.width - dw) / 2f
    val top = (into.height - dh) / 2f
    return androidx.compose.ui.geometry.Rect(left, top, left + dw, top + dh)
}

/** Transparency grid, so cut-outs stay visible like in the desktop viewer. */
private fun DrawScope.drawCheckerboard(area: Size) {
    val cell = 12.dp.toPx()
    drawRect(color = Color(0xFF343434), size = area)
    var y = 0f
    var row = 0
    while (y < area.height) {
        var x = 0f
        var col = 0
        while (x < area.width) {
            if ((row + col) % 2 == 0) {
                drawRect(
                    color = Color(0xFF484848),
                    topLeft = Offset(x, y),
                    size = Size(minOf(cell, area.width - x), minOf(cell, area.height - y)),
                )
            }
            x += cell
            col++
        }
        y += cell
        row++
    }
}
