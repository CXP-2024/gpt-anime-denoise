package com.animedenoise.image

import android.content.ContentResolver
import android.content.ContentValues
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * Saves results where the user expects them: either overwriting the picked file, or a new file in
 * Pictures/AnimeDenoise via MediaStore so no storage permission is needed on any supported version.
 */
object Persistence {

    private const val TAG = "Persistence"
    private const val FOLDER = "AnimeDenoise"

    /** Writes a PNG next to the gallery, returning its display URI. */
    fun saveToPictures(resolver: ContentResolver, name: String, png: ByteArray): Uri? {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/png")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/" + FOLDER)
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
        }
        val collection = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            MediaStore.Images.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        } else {
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        }
        val uri = resolver.insert(collection, values) ?: return null
        try {
            resolver.openOutputStream(uri)?.use { it.write(png) } ?: return null
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                resolver.update(uri, ContentValues().apply { put(MediaStore.MediaColumns.IS_PENDING, 0) }, null, null)
            }
            return uri
        } catch (t: Throwable) {
            Log.w(TAG, "save failed", t)
            runCatching { resolver.delete(uri, null, null) }
            return null
        }
    }

    /**
     * Overwrites the user's own file through the granted URI.
     *
     * Picking a document usually grants write access to that one document, which is exactly what is needed;
     * a failure is reported rather than silently dropping the result.
     */
    fun overwrite(resolver: ContentResolver, uri: Uri, png: ByteArray): Boolean {
        return try {
            val stream = resolver.openOutputStream(uri, "wt") ?: return false
            stream.use { it.write(png) }
            true
        } catch (t: Throwable) {
            Log.w(TAG, "overwrite failed", t)
            false
        }
    }

    /** App-private copy, used for the share sheet and for comparing runs. */
    fun saveLocal(dir: File, name: String, png: ByteArray): File {
        dir.mkdirs()
        val file = File(dir, name)
        FileOutputStream(file).use { it.write(png) }
        return file
    }

    fun suggestName(sourceName: String?, method: String): String {
        val stem = (sourceName ?: "image")
            .substringBeforeLast('.', sourceName ?: "image")
            .replace(Regex("[\\\\/:*?\"<>|\\x00-\\x1f]"), "_")
            .trim()
            .take(90)
            .ifBlank { "image" }
        return stem + "_" + method + ".png"
    }
}
