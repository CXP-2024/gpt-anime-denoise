package com.animedenoise

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log

/**
 * Runs the verification long enough to finish.
 *
 * A plain BroadcastReceiver is the wrong host for work that takes seconds: as soon as onReceive returns
 * the process is moved to "background->idle" and Android freezes or throttles it, which showed up as the
 * self test stalling mid-run on MIUI. A foreground service with a live notification keeps the process in
 * an active state until the run genuinely completes.
 */
class SelfTestService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, notification())
        Thread({
            try {
                SelfTest.attach(applicationContext)
                SelfTest.run(applicationContext)
            } catch (t: Throwable) {
                Log.e(TAG, "self test crashed", t)
            } finally {
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }, "selftest").start()
        return START_NOT_STICKY
    }

    private fun notification(): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL, "自检", NotificationManager.IMPORTANCE_LOW,
            ).apply { description = "像素级自检运行时保持进程活跃" }
            manager.createNotificationChannel(channel)
        }
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }
        return builder
            .setContentTitle("正在自检")
            .setContentText("对比桌面版参考输出，请稍候")
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setOngoing(true)
            .build()
    }

    private companion object {
        const val TAG = "SelfTest"
        const val CHANNEL = "selftest"
        const val NOTIFICATION_ID = 4242
    }
}
