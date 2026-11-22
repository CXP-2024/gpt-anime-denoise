package com.animedenoise

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log

/**
 * Entry point for the on-device verification:
 *
 *   adb shell am broadcast -a com.animedenoise.SELFTEST -n com.animedenoise/.SelfTestReceiver
 *
 * The work is delegated to a foreground service because a receiver's process is frozen once onReceive
 * returns, which is far sooner than the test finishes.
 */
class SelfTestReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        Log.i("SelfTest", "broadcast received, starting service")
        val service = Intent(context, SelfTestService::class.java)
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            context.startForegroundService(service)
        } else {
            context.startService(service)
        }
    }
}
