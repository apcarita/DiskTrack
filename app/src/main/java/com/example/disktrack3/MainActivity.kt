package com.example.disktrack3

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
import com.google.android.gms.tflite.java.TfLite
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

class MainActivity : ComponentActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private val TAG = "MainActivity"
    private val CAMERA_PERMISSION_REQUEST = 100

    private lateinit var cameraView: JavaCameraView
    private var mRgba: Mat? = null
    private var grayFrameForTracker: Mat? = null // Pre-allocate Mat for grayscale conversion

    private var processingMode = 0
    private val TOTAL_MODES = 6 // HOG removed, YOLO added, Tracker added

    @Volatile private var yoloDetector: YoloDetector? = null
    private val YOLO_CPU_THREADS = 4

    private lateinit var cannyEdgeProcessor: CannyEdgeProcessor

    private lateinit var yoloTracker: YoloTracker
    private val currentTracks = AtomicReference<List<Pair<Int, Rect>>>(emptyList())

    private lateinit var backgroundExecutor: ExecutorService
    private val isProcessingFrame = AtomicBoolean(false)
    private val latestDetections = AtomicReference<List<YoloDetector.Detection>?>(null)

    private var yoloFrameCount: Int = 0
    private var lastYoloFpsTime: Long = 0
    private var currentYoloFps: Double = 0.0
    private lateinit var yoloFpsTextView: TextView
    private val YOLO_FPS_UPDATE_INTERVAL_MS = 1000L

    private val rectColor = Scalar(0.0, 255.0, 0.0)
    private val textColor = Scalar(255.0, 255.0, 255.0)
    private val infoColor = Scalar(255.0, 0.0, 0.0)
    private val trackIdColor = Scalar(255.0, 255.0, 0.0)

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                Log.d(TAG, "Permission granted by user.")
                cameraView.setCameraPermissionGranted()
                cameraView.enableView()
            } else {
                Log.d(TAG, "Permission denied by user.")
                Toast.makeText(this, "Camera permission is required to use this app.", Toast.LENGTH_LONG).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "onCreate: START")

        setContentView(R.layout.activity_main)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed")
            Toast.makeText(this, "OpenCV initialization failed", Toast.LENGTH_LONG).show()
            finish()
        } else {
            Log.d(TAG, "OpenCV loaded successfully")
        }

        cameraView = findViewById(R.id.camera_view)
        cameraView.visibility = SurfaceView.VISIBLE
        try {
            cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK)
            Log.d(TAG, "Using back camera")
        } catch (e: Exception) {
            Log.e(TAG, "Error setting back camera, trying front camera", e)
            try {
                cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT)
                Log.d(TAG, "Using front camera")
            } catch (e: Exception) {
                Log.e(TAG, "Error setting front camera", e)
            }
        }
        cameraView.setMaxFrameSize(640, 480)
        cameraView.setCvCameraViewListener(this)

        yoloFpsTextView = findViewById(R.id.yolo_fps_text)

        cameraView.setOnClickListener {
            processingMode = (processingMode + 1) % TOTAL_MODES
            val modeName = when (processingMode) {
                0 -> "Normal"
                1 -> "Grayscale"
                2 -> "Canny Edge Detection"
                3 -> "Color Inversion"
                4 -> "YOLO Object Detection"
                5 -> "YOLO + Tracking"
                else -> "Unknown"
            }
            if (processingMode != 5) {
                yoloTracker.clear()
                currentTracks.set(emptyList())
            }
            Toast.makeText(this, "Mode: $modeName", Toast.LENGTH_SHORT).show()
        }

        backgroundExecutor = Executors.newSingleThreadExecutor()

        initializeTfLiteAndDetector()

        cannyEdgeProcessor = CannyEdgeProcessor()

        yoloTracker = YoloTracker()
        Log.d(TAG, "onCreate: YoloTracker initialized.")

        requestCameraPermission()
        Log.d(TAG, "onCreate: END")
    }

    private fun initializeTfLiteAndDetector() {
        TfLite.initialize(this)
            .addOnSuccessListener {
                Log.i(TAG, "TfLite initialized successfully.")
                createYoloDetector()
            }
            .addOnFailureListener { e: Exception ->
                Log.e(TAG, "TfLite initialization failed.", e)
                runOnUiThread {
                    Toast.makeText(this, "TFLite initialization failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
    }

    private fun createYoloDetector() {
        try {
            backgroundExecutor.submit {
                try {
                    val options = InterpreterApi.Options()
                        .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
                        .setNumThreads(YOLO_CPU_THREADS)

                    yoloDetector = YoloDetector(this, options)
                    Log.w(TAG, "YoloDetector created. Ready: ${yoloDetector?.isReady()}")
                } catch (e: Exception) {
                    Log.e(TAG, "Error creating YoloDetector: ${e.message}", e)
                }
            }
        } catch (submitError: Exception) {
            Log.e(TAG, "Error submitting task for YoloDetector creation: ${submitError.message}", submitError)
        }
    }

    private fun requestCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                Log.d(TAG, "Camera permission already granted.")
                cameraView.setCameraPermissionGranted()
            }
            shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {
                Log.d(TAG, "Showing permission rationale.")
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
            else -> {
                Log.d(TAG, "Requesting camera permission.")
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            Log.d(TAG, "onResume: Camera permission granted, enabling view.")
            cameraView.enableView()
        } else {
            Log.d(TAG, "onResume: Camera permission not granted.")
        }
    }

    override fun onPause() {
        super.onPause()
        cameraView.disableView()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraView.disableView()
        yoloTracker.clear()

        backgroundExecutor.submit {
            try {
                yoloDetector?.close()
            } catch (e: Exception) {
                Log.e(TAG, "Error closing YoloDetector", e)
            }
        }

        backgroundExecutor.shutdown()
        try {
            if (!backgroundExecutor.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)) {
                backgroundExecutor.shutdownNow()
            }
        } catch (e: InterruptedException) {
            backgroundExecutor.shutdownNow()
            Thread.currentThread().interrupt()
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        mRgba = Mat(height, width, CvType.CV_8UC4)
        grayFrameForTracker = Mat(height, width, CvType.CV_8UC1) // Initialize grayscale Mat
        lastYoloFpsTime = System.currentTimeMillis()
        yoloTracker.clear()
        currentTracks.set(emptyList())
    }

    override fun onCameraViewStopped() {
        mRgba?.release()
        grayFrameForTracker?.release() // Release grayscale Mat
        grayFrameForTracker = null
        yoloTracker.clear()
        currentTracks.set(emptyList())
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        val currentFrame = inputFrame.rgba()
        var displayFrame = currentFrame // Use original RGBA frame for display
        val detectionsForTracker: List<YoloDetector.Detection>? = latestDetections.getAndSet(null)

        try {
            when (processingMode) {
                0 -> { /* Normal */ }
                1 -> {
                    val grayFrame = Mat()
                    Imgproc.cvtColor(currentFrame, grayFrame, Imgproc.COLOR_RGBA2GRAY)
                    Imgproc.cvtColor(grayFrame, grayFrame, Imgproc.COLOR_GRAY2RGBA)
                    displayFrame = grayFrame
                }
                2 -> {
                    displayFrame = cannyEdgeProcessor.processFrame(currentFrame)
                }
                3 -> {
                    val invertedFrame = Mat()
                    Core.bitwise_not(currentFrame, invertedFrame)
                    displayFrame = invertedFrame
                }
                4, 5 -> {
                    val currentDetector = yoloDetector
                    if (currentDetector != null && currentDetector.isReady() && !isProcessingFrame.get()) {
                        if (isProcessingFrame.compareAndSet(false, true)) {
                            val frameCopy = currentFrame.clone()

                            backgroundExecutor.submit {
                                try {
                                    val frameWidth = frameCopy.cols()
                                    val frameHeight = frameCopy.rows()

                                    val detections = currentDetector.detect(frameCopy, frameWidth, frameHeight)
                                    latestDetections.set(detections)

                                    yoloFrameCount++
                                    val now = System.currentTimeMillis()
                                    val elapsed = now - lastYoloFpsTime
                                    if (elapsed >= YOLO_FPS_UPDATE_INTERVAL_MS) {
                                        currentYoloFps = yoloFrameCount * 1000.0 / elapsed
                                        lastYoloFpsTime = now
                                        yoloFrameCount = 0
                                        runOnUiThread {
                                            yoloFpsTextView.text = String.format(java.util.Locale.US, "YOLO FPS: %.2f", currentYoloFps)
                                        }
                                    }
                                } catch (e: Exception) {
                                    latestDetections.set(emptyList())
                                } finally {
                                    frameCopy.release()
                                    isProcessingFrame.set(false)
                                }
                            }
                        }
                    }

                    if (processingMode == 5) {
                        if (grayFrameForTracker != null) {
                            Imgproc.cvtColor(currentFrame, grayFrameForTracker, Imgproc.COLOR_RGBA2GRAY)

                            if (grayFrameForTracker!!.empty() || grayFrameForTracker!!.type() != CvType.CV_8UC1) {
                                Log.e(TAG, "Invalid grayFrameForTracker! empty=${grayFrameForTracker!!.empty()}, type=${CvType.typeToString(grayFrameForTracker!!.type())}, expected=${CvType.typeToString(CvType.CV_8UC1)}")
                                currentTracks.set(emptyList())
                            } else {
                                Log.d(TAG, "Passing gray frame to tracker: Size=${grayFrameForTracker!!.size()}, Type=${CvType.typeToString(grayFrameForTracker!!.type())}, Detections=${detectionsForTracker?.size ?: "null"}")

                                val tracks = try {
                                    yoloTracker.update(grayFrameForTracker!!, detectionsForTracker)
                                } catch (trackerUpdateError: Exception) {
                                    Log.e(TAG, "CRITICAL: Error calling yoloTracker.update: ${trackerUpdateError.message}", trackerUpdateError)
                                    emptyList<Pair<Int, Rect>>()
                                }
                                Log.d(TAG, "yoloTracker.update returned ${tracks.size} tracks.")
                                currentTracks.set(tracks)
                            }
                        } else {
                            Log.e(TAG, "grayFrameForTracker is null! Skipping tracker update.")
                            currentTracks.set(emptyList())
                        }

                        val tracksToDraw = currentTracks.get()
                        if (tracksToDraw.isNotEmpty()) {
                            for ((id, rect) in tracksToDraw) {
                                Imgproc.rectangle(displayFrame, rect, rectColor, 2)
                                val label = "ID: $id"
                                Imgproc.putText(
                                    displayFrame, label, Point(rect.x.toDouble(), rect.y.toDouble() - 10),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, trackIdColor, 2
                                )
                            }
                            Imgproc.putText(
                                displayFrame,
                                "Tracking: ${tracksToDraw.size}",
                                Point(10.0, 60.0),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, infoColor, 2
                            )
                        } else {
                            Imgproc.putText(
                                displayFrame, "No active tracks", Point(10.0, 60.0),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, infoColor, 2
                            )
                        }

                        if (detectionsForTracker != null && detectionsForTracker.isNotEmpty()) {
                            Imgproc.putText(
                                displayFrame,
                                "YOLO Update: ${detectionsForTracker.size}",
                                Point(10.0, 30.0),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2
                            )
                        } else if (isProcessingFrame.get()) {
                            Imgproc.putText(
                                displayFrame, "YOLO Processing...", Point(10.0, 30.0),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2
                            )
                        } else {
                            Imgproc.putText(
                                displayFrame, "Tracking...", Point(10.0, 30.0),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2
                            )
                        }

                    } else {
                        val detectionsToDraw = detectionsForTracker ?: latestDetections.get() ?: emptyList()
                        if (detectionsToDraw.isNotEmpty()) {
                            for (det in detectionsToDraw) {
                                Imgproc.rectangle(displayFrame, det.boundingBox, rectColor, 2)
                                val label = "${det.className} ${String.format("%.2f", det.confidence)}"
                                Imgproc.putText(
                                    displayFrame, label, Point(det.boundingBox.x.toDouble(), det.boundingBox.y.toDouble() - 10),
                                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2
                                )
                            }
                            Imgproc.putText(
                                displayFrame,
                                "Detections: ${detectionsToDraw.size}",
                                Point(10.0, 30.0),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, infoColor, 2
                            )
                        } else if (yoloDetector == null) {
                            Imgproc.putText(displayFrame, "Initializing Detector...", Point(10.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2)
                        } else if (isProcessingFrame.get()) {
                            Imgproc.putText(displayFrame, "Processing...", Point(10.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, infoColor, 2)
                        } else {
                            Imgproc.putText(displayFrame, "No detections", Point(10.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, infoColor, 2)
                        }
                        currentTracks.set(emptyList())
                    }

                    runOnUiThread {
                        yoloFpsTextView.visibility = View.VISIBLE
                    }
                }
                else -> { }
            }

            if (processingMode != 4 && processingMode != 5) {
                runOnUiThread {
                    yoloFpsTextView.visibility = View.GONE
                }
            }

        } catch (e: Exception) {
            runOnUiThread { yoloFpsTextView.visibility = View.GONE }
            return currentFrame
        }

        return displayFrame
    }
}