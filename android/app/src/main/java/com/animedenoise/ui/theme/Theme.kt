package com.animedenoise.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/**
 * The app is a photo tool, so both schemes are calm and low-contrast: chrome recedes, the image
 * carries the screen. Violet accents tie the UI to the launcher icon.
 */
private val Light = lightColorScheme(
    primary = Color(0xFF5B4BC4),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFE4DEFF),
    onPrimaryContainer = Color(0xFF1B1160),
    secondary = Color(0xFF4F4C66),
    onSecondary = Color.White,
    surface = Color(0xFFFDFCFF),
    onSurface = Color(0xFF1B1B21),
    surfaceVariant = Color(0xFFE7E3F2),
    onSurfaceVariant = Color(0xFF49455A),
    outline = Color(0xFF79748F),
    background = Color(0xFFF6F4FA),
    onBackground = Color(0xFF1B1B21),
    error = Color(0xFFBA1A1A),
)

private val Dark = darkColorScheme(
    primary = Color(0xFFC0B3FF),
    onPrimary = Color(0xFF271D78),
    primaryContainer = Color(0xFF3D2F97),
    onPrimaryContainer = Color(0xFFE4DEFF),
    secondary = Color(0xFFC6C3DD),
    onSecondary = Color(0xFF302E47),
    surface = Color(0xFF14121B),
    onSurface = Color(0xFFE6E1EF),
    surfaceVariant = Color(0xFF211E2B),
    onSurfaceVariant = Color(0xFFC9C4D8),
    outline = Color(0xFF49455A),
    background = Color(0xFF0F0E14),
    onBackground = Color(0xFFE6E1EF),
    error = Color(0xFFFFB4AB),
)

@Composable
fun AnimeDenoiseTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    val colors = if (darkTheme) Dark else Light
    MaterialTheme(colorScheme = colors, content = content)
}
