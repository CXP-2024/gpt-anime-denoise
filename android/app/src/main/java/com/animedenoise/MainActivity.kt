package com.animedenoise

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.core.content.IntentCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.animedenoise.ui.DenoiseScreen
import com.animedenoise.ui.DenoiseViewModel
import com.animedenoise.ui.theme.AnimeDenoiseTheme

class MainActivity : ComponentActivity() {

    private var pendingUri: ((Uri?) -> Unit)? = null

    /** The most recent image handed to us by a SEND or VIEW intent. */
    private val incoming = mutableStateOf<Uri?>(null)

    private val picker = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        val callback = pendingUri
        pendingUri = null
        callback?.invoke(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // adb shell am start -n com.animedenoise/.MainActivity --ez selftest true
        if (intent?.getBooleanExtra(EXTRA_SELFTEST, false) == true) {
            SelfTest.attach(applicationContext)
            SelfTest.run(applicationContext)
        }
        incoming.value = intent?.toImageUri()
        setContent {
            AnimeDenoiseTheme {
                val viewModel: DenoiseViewModel = viewModel()
                // When the selftest runs (--ez selftest true), let it drive the visible UI so the
                // user sees real progress instead of a frozen screen.
                com.animedenoise.SelfTest.uiListener = { source ->
                    viewModel.loadDirect(source, "selftest")
                }
                val shared by incoming
                androidx.compose.runtime.LaunchedEffect(shared) {
                    shared?.let { viewModel.pick(it, displayNameOf(it)) }
                }
                Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    DenoiseScreen(
                        viewModel = viewModel,
                        onPickImage = { callback ->
                            pendingUri = callback
                            picker.launch(
                                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                            )
                        },
                    )
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        if (intent.getBooleanExtra(EXTRA_SELFTEST, false)) {
            SelfTest.attach(applicationContext)
            SelfTest.run(applicationContext)
        }
        incoming.value = intent.toImageUri()
    }

    private companion object {
        const val EXTRA_SELFTEST = "selftest"
    }

    /** Real file name, so saved results are named after the picture instead of a content id. */
    private fun displayNameOf(uri: Uri): String? = try {
        contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
            ?: uri.lastPathSegment
    } catch (t: Throwable) {
        uri.lastPathSegment
    }
}

private fun Intent.toImageUri(): Uri? {
    val type = type ?: ""
    return when (action) {
        Intent.ACTION_SEND ->
            if (type.startsWith("image/") || type.isEmpty()) {
                IntentCompat.getParcelableExtra(this, Intent.EXTRA_STREAM, Uri::class.java)
            } else {
                null
            }
        Intent.ACTION_VIEW -> data
        else -> null
    }
}
