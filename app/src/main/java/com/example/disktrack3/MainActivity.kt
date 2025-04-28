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
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max

class MainActivity : ComponentActivity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private val TAG = "MainActivity"
    private val CAMERA_PERMISSION_REQUEST = 100

    private lateinit var cameraView: JavaCameraView
    private var mRgba: Mat? = null

    // Track processing mode for demo
    private var processingMode = 0
    private val TOTAL_MODES = 5 // HOG removed, YOLO added

    // --- YOLO Detection ---
    @Volatile private var yoloDetector: YoloDetector? = null // Make nullable and volatile for async init
    // --- End YOLO Detection ---

    // --- Canny Edge Processing ---
    private lateinit var cannyEdgeProcessor: CannyEdgeProcessor // Instance for Canny processing
    // --- End Canny Edge Processing ---

    // --- Background Processing ---
    private lateinit var backgroundExecutor: ExecutorService
    private val isProcessingFrame = AtomicBoolean(false)
    private val latestDetections = AtomicReference<List<YoloDetector.Detection>>(emptyList())
    // --- YOLO FPS Calculation ---
    private var yoloFrameCount: Int = 0
    private var lastYoloFpsTime: Long = 0
    private var currentYoloFps: Double = 0.0
    private lateinit var yoloFpsTextView: TextView // TextView for displaying YOLO FPS
    private val YOLO_FPS_UPDATE_INTERVAL_MS = 1000L // Update FPS display every second
    // --- End Background Processing ---

    // For bounding box drawing
    private val rectColor = Scalar(0.0, 255.0, 0.0) // Green color
    private val textColor = Scalar(255.0, 255.0, 255.0) // White color
    private val infoColor = Scalar(255.0, 0.0, 0.0) // Blue color for info text

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "onCreate: START") // Very first log

        Log.d(TAG, "onCreate: Starting app initialization")

        setContentView(R.layout.activity_main)

        // Keep screen on while using camera
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Check OpenCV initialization status
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed")
            Toast.makeText(this, "OpenCV initialization failed", Toast.LENGTH_LONG).show()
            finish()
        } else {
            Log.d(TAG, "OpenCV loaded successfully")
        }

        Log.d(TAG, "onCreate: Initializing CameraView...")
        // Initialize camera view
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
        Log.d(TAG, "Camera max frame size set to 640x480")
        cameraView.setCvCameraViewListener(this)
        Log.d(TAG, "onCreate: CameraView initialized.")

        // Initialize the YOLO FPS TextView
        Log.d(TAG, "onCreate: Finding YOLO FPS TextView...")
        yoloFpsTextView = findViewById(R.id.yolo_fps_text)

        // Set up touch listener to cycle through effects
        cameraView.setOnClickListener {
            processingMode = (processingMode + 1) % TOTAL_MODES
            val modeName = when (processingMode) {
                0 -> "Normal"
                1 -> "Grayscale"
                2 -> "Canny Edge Detection"
                3 -> "Color Inversion"
                4 -> "YOLO Object Detection" // Updated mode name
                else -> "Unknown"
            }
            Toast.makeText(this, "Mode: $modeName", Toast.LENGTH_SHORT).show()
        }

        // Initialize background executor EARLIER
        backgroundExecutor = Executors.newFixedThreadPool(2)
        Log.d(TAG, "onCreate: Background executor initialized with 2 threads.")

        // Directly create the detector using the new method
        createYoloDetector()

        // Initialize CannyEdgeProcessor
        cannyEdgeProcessor = CannyEdgeProcessor()

        requestCameraPermission()
        Log.d(TAG, "onCreate: END")
    }

    // Separate function to create the detector using standalone LiteRT
    private fun createYoloDetector() {
        backgroundExecutor.submit {
            Log.d(TAG, "BACKGROUND_TASK: Starting YoloDetector creation (CPU focus)...")
            var createdDetector: YoloDetector? = null
            var delegateUsed = "CPU (Default/XNNPACK)" // Assume default CPU

            try {
                // Create detector directly, letting it handle options
                createdDetector = YoloDetector(applicationContext)

                if (createdDetector.isReady()) {
                    Log.i(TAG, "YoloDetector created successfully with default CPU options.")
                } else {
                    Log.e(TAG, "YoloDetector failed to initialize even with default CPU.")
                    createdDetector = null // Ensure detector is null
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception creating YoloDetector with default CPU options", e)
                createdDetector = null // Ensure detector is null
            }

            // Assign the successfully created detector (or null if all failed)
            yoloDetector = createdDetector

            Log.d(TAG, "BACKGROUND_TASK: YoloDetector creation finished. Delegate Used: $delegateUsed. Final Ready State: ${yoloDetector?.isReady() ?: false}")
        }
    }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST
            )
        } else {
            Log.d(TAG, "Camera permission already granted")
            cameraView.setCameraPermissionGranted()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.d(TAG, "Camera permission granted")
                cameraView.setCameraPermissionGranted()
                if (cameraView != null) cameraView.enableView()
            } else {
                Log.d(TAG, "Camera permission denied")
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            Log.d(TAG, "onResume: Enabling camera view")
            if (cameraView != null) {
                cameraView.enableView()
            }
        } else {
            Log.d(TAG, "onResume: Camera permission not granted yet")
        }
    }

    override fun onPause() {
        super.onPause()
        Log.d(TAG, "onPause: Disabling camera view")
        if (cameraView != null) {
            cameraView.disableView()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "onDestroy: Disabling camera view, shutting down executor, and closing detector")
        if (cameraView != null) {
            cameraView.disableView()
        }

        backgroundExecutor.submit {
            try {
                Log.d(TAG, "onDestroy Background Task: Closing YoloDetector...")
                yoloDetector?.close()
                Log.d(TAG, "onDestroy Background Task: YoloDetector close requested.")
            } catch (e: Exception) {
                Log.e(TAG, "onDestroy Background Task: Error closing YoloDetector", e)
            }
        }

        backgroundExecutor.shutdown()
        try {
            Log.d(TAG, "onDestroy: Awaiting background executor termination...")
            if (!backgroundExecutor.awaitTermination(3, java.util.concurrent.TimeUnit.SECONDS)) {
                Log.w(TAG, "onDestroy: Background executor did not terminate in 3 seconds.")
                backgroundExecutor.shutdownNow()
            } else {
                Log.d(TAG, "onDestroy: Background executor terminated gracefully.")
            }
        } catch (e: InterruptedException) {
            Log.e(TAG, "onDestroy: Interrupted while awaiting executor termination.", e)
            backgroundExecutor.shutdownNow()
            Thread.currentThread().interrupt()
        }

        Log.d(TAG, "onDestroy: Finished.")
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        Log.d(TAG, "onCameraViewStarted: width=$width, height=$height")
        mRgba = Mat(height, width, CvType.CV_8UC4)
        lastYoloFpsTime = System.currentTimeMillis()
    }

    override fun onCameraViewStopped() {
        Log.d(TAG, "onCameraViewStopped")
        mRgba?.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        val currentFrame = inputFrame.rgba()
        var hideFps = true

        try {
            when (processingMode) {
                0 -> {
                    return currentFrame
                }
                1 -> {
                    val grayFrame = Mat()
                    Imgproc.cvtColor(currentFrame, grayFrame, Imgproc.COLOR_RGBA2GRAY)
                    Imgproc.cvtColor(grayFrame, grayFrame, Imgproc.COLOR_GRAY2RGBA)
                    return grayFrame
                }
                2 -> {
                    val edgesResult = cannyEdgeProcessor.processFrame(currentFrame)
                    return edgesResult
                }
                3 -> {
                    val invertedFrame = Mat()
                    Core.bitwise_not(currentFrame, invertedFrame)
                    return invertedFrame
                }
                4 -> {
                    hideFps = false
                    if (yoloDetector == null) {
                        Log.w(TAG, "onCameraFrame: yoloDetector is still null.")
                    }

                    val currentDetector = yoloDetector
                    if (currentDetector != null && currentDetector.isReady() && !isProcessingFrame.get()) {
                        if (isProcessingFrame.compareAndSet(false, true)) {
                            val frameCopy = currentFrame.clone()

                            backgroundExecutor.submit {
                                var bitmap: Bitmap? = null
                                try {
                                    val startTime = System.currentTimeMillis()
                                    val frameWidth = frameCopy.cols()
                                    val frameHeight = frameCopy.rows()

                                    bitmap = Bitmap.createBitmap(frameWidth, frameHeight, Bitmap.Config.ARGB_8888)
                                    Utils.matToBitmap(frameCopy, bitmap)

                                    val detections = currentDetector.detect(bitmap, frameWidth, frameHeight)
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

                                    val endTime = System.currentTimeMillis()
                                    Log.d(TAG, "Background Detection Time: ${endTime - startTime} ms, Found: ${detections.size}")

                                } catch (e: Exception) {
                                    Log.e(TAG, "Error in background YOLO processing: ${e.message}", e)
                                    latestDetections.set(emptyList())
                                } finally {
                                    bitmap?.recycle()
                                    frameCopy.release()
                                    isProcessingFrame.set(false)
                                }
                            }
                        }
                    }

                    val detectionsToDraw = latestDetections.get()
                    if (detectionsToDraw.isNotEmpty()) {
                        for (det in detectionsToDraw) {
                            Imgproc.rectangle(currentFrame, det.boundingBox, rectColor, 2)
                            val label = "${det.className} ${String.format("%.2f", det.confidence)}"
                            Imgproc.putText(
                                currentFrame, label, Point(det.boundingBox.x.toDouble(), det.boundingBox.y.toDouble() - 10),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2
                            )
                        }
                        Imgproc.putText(
                            currentFrame,
                            "Drawn: ${detectionsToDraw.size} (Prev)",
                            Point(10.0, 30.0),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, infoColor, 2
                        )
                    } else if (yoloDetector == null) {
                        Imgproc.putText(
                            currentFrame, "Initializing Detector...", Point(10.0, 30.0),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2
                        )
                    } else if (isProcessingFrame.get()) {
                        Imgproc.putText(
                            currentFrame, "Processing...", Point(10.0, 30.0),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, infoColor, 2
                        )
                    }

                    return currentFrame
                }
                else -> {
                    return currentFrame
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in frame processing: ${e.message}", e)
            hideFps = true
            return currentFrame
        } finally {
            runOnUiThread {
                yoloFpsTextView.visibility = if (hideFps) View.GONE else View.VISIBLE
            }
        }
    }
}