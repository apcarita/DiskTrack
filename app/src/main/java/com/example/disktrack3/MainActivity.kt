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
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
import com.google.android.gms.tflite.java.TfLite // Correct import for Play Services TFLite initialization
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

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
    private val YOLO_CPU_THREADS = 4 // Define number of threads for CPU inference
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
        backgroundExecutor = Executors.newSingleThreadExecutor()
        Log.d(TAG, "onCreate: Background executor initialized.")

        // Initialize LiteRT and YoloDetector asynchronously
        initializeTfLiteAndDetector() // Call the initialization function

        // Initialize CannyEdgeProcessor
        cannyEdgeProcessor = CannyEdgeProcessor()

        requestCameraPermission()
        Log.d(TAG, "onCreate: END")
    }

    // Function to initialize Play Services TFLite and the detector
    private fun initializeTfLiteAndDetector() {
        Log.d(TAG, "INITIALIZE: Starting TfLite initialization...")
        TfLite.initialize(this)
            .addOnSuccessListener {
                Log.i(TAG, "INITIALIZE: TfLite (Play Services) initialized successfully.")
                // Now that TfLite is initialized, proceed to create the detector on the background thread
                createYoloDetector()
            }
            .addOnFailureListener { e: Exception -> // Explicitly type the exception parameter
                Log.e(TAG, "INITIALIZE: TfLite (Play Services) initialization failed.", e)
                // Handle initialization failure (e.g., show a message to the user)
                runOnUiThread {
                    Toast.makeText(this, "TFLite initialization failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
    }

    // Separate function to create the detector after TfLite initialization succeeds
    private fun createYoloDetector() {
        try {
            backgroundExecutor.submit {
                Log.d(TAG, "BACKGROUND_TASK: Task submitted to executor has STARTED (createYoloDetector).")
                Log.w(TAG, "BACKGROUND_TASK: Attempting to create YoloDetector (CPU via Play Services with $YOLO_CPU_THREADS threads)...")
                try {
                    // Create options for CPU execution using Play Services Runtime
                    val options = InterpreterApi.Options()
                        .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY) // Use Play Services
                        .setNumThreads(YOLO_CPU_THREADS) // Set the number of CPU threads

                    Log.i(TAG, "BACKGROUND_TASK: Interpreter options created (CPU, $YOLO_CPU_THREADS threads).")

                    // Initialize the detector with CPU options
                    yoloDetector = YoloDetector(this, options) // Pass context and configured options
                    Log.w(TAG, "BACKGROUND_TASK: YoloDetector (CPU) created. Ready: ${yoloDetector?.isReady()}")

                } catch (e: Exception) {
                    Log.e(TAG, "BACKGROUND_TASK: CRITICAL Error creating YoloDetector (CPU): ${e.message}", e)
                }
            }
            Log.d(TAG, "INITIALIZE: backgroundExecutor.submit() call for createYoloDetector completed.")
        } catch (submitError: Exception) {
            Log.e(TAG, "INITIALIZE: CRITICAL Error calling backgroundExecutor.submit() for createYoloDetector: ${submitError.message}", submitError)
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

        // Submit the close operation to the background thread
        backgroundExecutor.submit {
            try {
                Log.d(TAG, "onDestroy Background Task: Closing YoloDetector...")
                yoloDetector?.close() // Close the detector if it was initialized
                Log.d(TAG, "onDestroy Background Task: YoloDetector close requested.")
            } catch (e: Exception) {
                Log.e(TAG, "onDestroy Background Task: Error closing YoloDetector", e)
            }
        }

        // Initiate shutdown, allowing submitted tasks (like the close task) to complete
        backgroundExecutor.shutdown()
        try {
            Log.d(TAG, "onDestroy: Awaiting background executor termination...")
            if (!backgroundExecutor.awaitTermination(5, java.util.concurrent.TimeUnit.SECONDS)) {
                Log.w(TAG, "onDestroy: Background executor did not terminate in 5 seconds.")
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
        lastYoloFpsTime = System.currentTimeMillis() // Initialize FPS timer
    }

    override fun onCameraViewStopped() {
        Log.d(TAG, "onCameraViewStopped")
        mRgba?.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        // Get the original frame
        val currentFrame = inputFrame.rgba() // RGBA format from camera
        var displayFrame = currentFrame // Frame to display, might be modified

        try {
            // Process frame based on selected mode
            when (processingMode) {
                0 -> { // Normal
                    // No processing needed
                }
                1 -> { // Grayscale
                    val grayFrame = Mat()
                    Imgproc.cvtColor(currentFrame, grayFrame, Imgproc.COLOR_RGBA2GRAY)
                    Imgproc.cvtColor(grayFrame, grayFrame, Imgproc.COLOR_GRAY2RGBA) // Convert back for display
                    displayFrame = grayFrame // Update display frame
                }
                2 -> { // Canny Edge Detection (Refactored)
                    displayFrame = cannyEdgeProcessor.processFrame(currentFrame) // Update display frame
                }
                3 -> { // Color Inversion
                    val invertedFrame = Mat()
                    Core.bitwise_not(currentFrame, invertedFrame)
                    displayFrame = invertedFrame // Update display frame
                }
                4 -> { // YOLO Object Detection (Background Thread)
                    Log.d(TAG, "onCameraFrame: Mode 4 (YOLO). Detector null? ${yoloDetector == null}. Ready? ${yoloDetector?.isReady()}. Processing? ${isProcessingFrame.get()}")

                    val currentDetector = yoloDetector // Local ref for thread safety
                    if (currentDetector != null && currentDetector.isReady() && !isProcessingFrame.get()) {
                        Log.d(TAG, "onCameraFrame: Conditions met, attempting to submit frame.")
                        if (isProcessingFrame.compareAndSet(false, true)) {
                            Log.d(TAG, "Submitting frame for YOLO processing")
                            val frameCopy = currentFrame.clone() // Clone the frame for background processing

                            backgroundExecutor.submit {
                                try {
                                    val startTime = System.currentTimeMillis()
                                    val frameWidth = frameCopy.cols()
                                    val frameHeight = frameCopy.rows()

                                    // --- Run Detection directly on the Mat ---
                                    // No Bitmap conversion needed here
                                    val detections = currentDetector.detect(frameCopy, frameWidth, frameHeight) // Pass Mat directly
                                    latestDetections.set(detections)
                                    // --- End Detection ---

                                    // --- Calculate YOLO FPS ---
                                    yoloFrameCount++
                                    val now = System.currentTimeMillis()
                                    val elapsed = now - lastYoloFpsTime
                                    if (elapsed >= YOLO_FPS_UPDATE_INTERVAL_MS) {
                                        currentYoloFps = yoloFrameCount * 1000.0 / elapsed
                                        lastYoloFpsTime = now
                                        yoloFrameCount = 0
                                        // Update FPS on UI thread inside the interval check
                                        runOnUiThread {
                                            yoloFpsTextView.text = String.format(java.util.Locale.US, "YOLO FPS: %.2f", currentYoloFps)
                                            yoloFpsTextView.visibility = View.VISIBLE
                                        }
                                    }
                                    // --- End YOLO FPS Calculation ---

                                    val endTime = System.currentTimeMillis()
                                    Log.d(TAG, "Background Detection Time: ${endTime - startTime} ms, Found: ${detections.size}")

                                    // No bitmap to recycle

                                } catch (e: Exception) {
                                    Log.e(TAG, "Error in background YOLO processing: ${e.message}", e)
                                    latestDetections.set(emptyList())
                                } finally {
                                    frameCopy.release() // Release the cloned Mat
                                    isProcessingFrame.set(false)
                                    Log.d(TAG, "Background processing finished.")
                                }
                            }
                        } else {
                             Log.d(TAG, "Skipping frame submission: Already processing or detector not ready.")
                        }
                    } else {
                         Log.d(TAG, "Skipping frame submission: Detector not ready or null.")
                    }

                    // --- Draw Latest Results on the original frame ---
                    val detectionsToDraw = latestDetections.get()
                    if (detectionsToDraw.isNotEmpty()) {
                        for (det in detectionsToDraw) {
                            Imgproc.rectangle(displayFrame, det.boundingBox, rectColor, 2) // Draw on displayFrame
                            val label = "${det.className} ${String.format("%.2f", det.confidence)}"
                            Imgproc.putText(
                                displayFrame, label, Point(det.boundingBox.x.toDouble(), det.boundingBox.y.toDouble() - 10),
                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2
                            )
                        }
                        Imgproc.putText(
                            displayFrame,
                            "Drawn: ${detectionsToDraw.size} (Prev)",
                            Point(10.0, 30.0),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, infoColor, 2
                        )
                    } else if (yoloDetector == null) { // Check if null due to async init
                        Imgproc.putText(
                            displayFrame, "Initializing Detector...", Point(10.0, 30.0),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2
                        )
                    } else if (isProcessingFrame.get()) {
                        Imgproc.putText(
                            displayFrame, "Processing...", Point(10.0, 30.0),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, infoColor, 2
                        )
                    } else {
                         // Optionally add text if no detections and not processing
                         Imgproc.putText(
                            displayFrame, "No detections", Point(10.0, 30.0),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, infoColor, 2
                        )
                    }

                    // Ensure YOLO FPS TextView is visible only in mode 4
                    runOnUiThread {
                        yoloFpsTextView.visibility = View.VISIBLE
                    }
                }
                else -> {
                    // No processing needed for other modes
                }
            }

            // Hide YOLO FPS TextView if not in YOLO mode (outside the when block)
            if (processingMode != 4) {
                 runOnUiThread {
                     yoloFpsTextView.visibility = View.GONE
                 }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error in frame processing: ${e.message}", e)
            // Hide YOLO FPS on error
            runOnUiThread {
                yoloFpsTextView.visibility = View.GONE
            }
            // Return original frame on error to prevent crash
            return currentFrame
        }

        // Return the potentially modified frame for display
        return displayFrame
    }
}