package com.animedenoise.sr

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log

/**
 * Hosts ExecuTorch GPU probing in its own process.
 *
 * Measured on a Redmi Note 12 Turbo (Adreno 725): a 128x128 Vulkan tile runs fine, a 384x384 tile
 * aborts natively inside Module.execute with "Abort message: 'ptr'" - a C++ null-check in the JNI
 * bridge that no Java try/catch can intercept. Any code that might exceed the GPU's tile budget must
 * therefore run where its death is cheap.
 */
class GpuProbeService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null
}
