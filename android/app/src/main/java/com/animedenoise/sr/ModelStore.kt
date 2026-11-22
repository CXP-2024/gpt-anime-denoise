package com.animedenoise.sr

import android.content.res.AssetManager
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * Access to the ONNX files shipped inside the APK.
 *
 * The models live at the root of assets/ (18 MB + 2.4 MB) and are stored uncompressed by aapt, so ORT can
 * memory-map them straight out of the APK instead of paying for a copy in RAM. [readBytes] is used because
 * ONNX Runtime's Kotlin API only takes a byte[] or a path; the caller releases the array right after
 * session creation, so the peak cost is one transient copy.
 */
object ModelStore {

    private const val TAG = "ModelStore"

    /** Files present in assets/, as bare names. */
    fun available(assets: AssetManager): Set<String> = try {
        assets.list("")?.filter { it.endsWith(".onnx") || it.endsWith(".pte") }?.toSet() ?: emptySet()
    } catch (t: Throwable) {
        Log.w(TAG, "asset listing failed", t)
        emptySet()
    }

    fun readBytes(assets: AssetManager, assetName: String): ByteArray =
        assets.open(assetName, AssetManager.ACCESS_BUFFER).use { it.readBytes() }

    fun has(assets: AssetManager, assetName: String): Boolean = try {
        assets.open(assetName, AssetManager.ACCESS_UNKNOWN).close()
        true
    } catch (e: java.io.IOException) {
        false
    }

    /**
     * Copies a model to app storage. Only needed for a path-based session, which we do not use today;
     * kept so a future QNN context-binary cache has somewhere to live.
     */
    fun ensureCached(assets: AssetManager, cacheDir: File, assetName: String): File {
        val target = File(cacheDir, assetName)
        if (target.isFile && target.length() > 0) return target
        cacheDir.mkdirs()
        assets.open(assetName, AssetManager.ACCESS_BUFFER).use { input ->
            FileOutputStream(target).use { output -> input.copyTo(output, 1 shl 16) }
        }
        return target
    }
}
