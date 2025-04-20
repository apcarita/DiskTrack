package com.example.disktrack3

import android.app.Application
import android.util.Log
import org.opencv.android.OpenCVLoader

class DiskTrackApplication : Application() {
    companion object {
        private const val TAG = "DiskTrackApplication"
    }

    override fun onCreate() {
        super.onCreate()
        
        // Initialize OpenCV at application startup
        try {
            if (!OpenCVLoader.initDebug()) {
                Log.e(TAG, "OpenCV initialization failed")
            } else {
                Log.d(TAG, "OpenCV initialization succeeded")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception during OpenCV initialization: ${e.message}")
            e.printStackTrace()
        }
    }
}
