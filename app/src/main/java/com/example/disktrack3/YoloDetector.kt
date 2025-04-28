package com.example.disktrack3

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfRect2d
import org.opencv.core.Rect
import org.opencv.core.Rect2d
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi

import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min

class YoloDetector(
    context: Context,
    options: InterpreterApi.Options
) {

    private val TAG = "YoloDetector"
    // --- Optimization: Use Quantized Model ---
    private val TFLITE_MODEL_FILE = "yolo11n.tflite" 
    // --- End Optimization ---

    private val YOLO_INPUT_SIZE = 640 // Input size remains 640 as requested
    private val YOLO_CONFIDENCE_THRESHOLD = 0.25f // Restore a reasonable threshold
    private val YOLO_IOU_THRESHOLD = 0.45f // Threshold for Non-Max Suppression

    private var tfliteInterpreter: InterpreterApi? = null
    private val isInitialized = AtomicBoolean(false)
    private var inputByteBuffer: ByteBuffer? = null
    private var outputBuffer: Array<Array<FloatArray>>? = null
    var classNames: List<String> = listOf()

    private val cocoLabels = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    )
    private val ACTUAL_EXPECTED_NUM_CLASSES = 80
    private val OUTPUT_NUM_PARAMS = 84 // 4 (bbox) + 80 (classes)
    private val OUTPUT_NUM_PREDICTIONS = 8400

    init {
        try {
            val assetManager = context.assets
            val modelBuffer = loadModelFile(assetManager, TFLITE_MODEL_FILE)

            // Initialize InterpreterApi using the provided options and factory method
            tfliteInterpreter = InterpreterApi.create(modelBuffer, options) // Use factory method

            // Buffer size needs to match the new YOLO_INPUT_SIZE
            val bufferSize = 1 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE * 3 * 4 // 3 channels (RGB), 4 bytes per float
            Log.d(TAG, "Allocating input buffer size: $bufferSize bytes for ${YOLO_INPUT_SIZE}x${YOLO_INPUT_SIZE}")
            inputByteBuffer = ByteBuffer.allocateDirect(bufferSize)
            inputByteBuffer?.order(ByteOrder.nativeOrder())

            val outputTensor = tfliteInterpreter?.getOutputTensor(0)

            val reportedOutputShape = outputTensor?.shape() ?: intArrayOf(0, 0, 0)
            val outputDataType = outputTensor?.dataType() ?: DataType.FLOAT32
            Log.d(TAG, "TFLite Reported Output Shape: ${reportedOutputShape.joinToString()}, DataType: $outputDataType")

            val BATCH_SIZE = 1
            Log.d(TAG, "Allocating output buffer with shape: [$BATCH_SIZE, $OUTPUT_NUM_PARAMS, $OUTPUT_NUM_PREDICTIONS]")
            outputBuffer = Array(BATCH_SIZE) { Array(OUTPUT_NUM_PARAMS) { FloatArray(OUTPUT_NUM_PREDICTIONS) } }

            val NUM_PARAMS = OUTPUT_NUM_PARAMS
            val NUM_CLASSES = NUM_PARAMS - 4
            Log.d(TAG, "Using FIXED Num Classes: $NUM_CLASSES")
            if (NUM_CLASSES != ACTUAL_EXPECTED_NUM_CLASSES) {
                Log.e(TAG, "FIXED class count ($NUM_CLASSES) does not match expected COCO count ($ACTUAL_EXPECTED_NUM_CLASSES). Check NUM_PARAMS.")
            }
            if (cocoLabels.size != ACTUAL_EXPECTED_NUM_CLASSES) {
                Log.w(TAG, "Mismatch between cocoLabels size (${cocoLabels.size}) and ACTUAL_EXPECTED_NUM_CLASSES ($ACTUAL_EXPECTED_NUM_CLASSES).")
            }

            classNames = cocoLabels

            isInitialized.set(true)
            Log.d(TAG, "YoloDetector initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing YoloDetector: ${e.message}", e)
            // InterpreterApi handles delegate closure internally on error or close()
            isInitialized.set(false)
        }
    }

    fun isReady(): Boolean = isInitialized.get()

    @Throws(Exception::class)
    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    data class Detection(val boundingBox: Rect, val confidence: Float, val classIndex: Int, val className: String)

    fun detect(rgbaMat: Mat, frameWidth: Int, frameHeight: Int): List<Detection> {
        if (!isInitialized.get() || tfliteInterpreter == null || inputByteBuffer == null || outputBuffer == null) {
            Log.w(TAG, "Detector not initialized or buffers are null.")
            return emptyList()
        }

        // --- Optimization 2: Alternative Preprocessing ---
        val resizedMat = Mat()
        val rgbMat = Mat()
        var floatMat = Mat() // Mat to hold float data - Make it var for potential cloning
        try {
            // 1. Resize the input Mat (RGBA)
            Imgproc.resize(rgbaMat, resizedMat, Size(YOLO_INPUT_SIZE.toDouble(), YOLO_INPUT_SIZE.toDouble()))

            // 2. Convert RGBA to RGB
            Imgproc.cvtColor(resizedMat, rgbMat, Imgproc.COLOR_RGBA2RGB)

            // 3. Convert RGB U8 to RGB Float32 and normalize
            rgbMat.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0)

            // 4. Populate ByteBuffer directly from float Mat
            inputByteBuffer?.rewind()
            val floatBuffer = inputByteBuffer?.asFloatBuffer()

            // --- Ensure Mat is continuous ---
            if (!floatMat.isContinuous) {
                 Log.w(TAG, "Cloning floatMat because it's not continuous.")
                 floatMat = floatMat.clone() // Clone to ensure continuous memory
            }
            // --- End Ensure Continuous ---

            // Get all float data at once
            val numElements = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE * 3
            val floatData = FloatArray(numElements)
            floatMat.get(0, 0, floatData) // Get float data from Mat

            // Put data into the buffer
            floatBuffer?.put(floatData)

        } catch (e: Exception) {
            Log.e(TAG, "Error during preprocessing: ${e.message}", e)
            return emptyList()
        } finally {
            // Release temporary Mats
            resizedMat.release()
            rgbMat.release()
            // floatMat is released even if it was cloned (original is unused after clone)
            floatMat.release()
        }
        // --- End Optimization 2 ---

        try {
            tfliteInterpreter?.run(inputByteBuffer, outputBuffer)
        } catch (e: Exception) {
            Log.e(TAG, "TFLite inference error: ${e.message}", e)
            return emptyList()
        }

        // --- Postprocessing (Keep only 'person' detections) ---
        val allDetections = processOutput(outputBuffer!!, frameWidth, frameHeight)
        // Filter for "person" class - case-insensitive comparison is safer
        val personDetections = allDetections.filter { it.className.equals("person", ignoreCase = true) }
        // Log if persons were detected vs total detections
        if (personDetections.isNotEmpty()) {
             Log.d(TAG, "Detected ${personDetections.size} persons out of ${allDetections.size} total detections.")
        }
        return personDetections
        // --- End Postprocessing ---
    }

    private fun processOutput(output: Array<Array<FloatArray>>, frameWidth: Int, frameHeight: Int): List<Detection> {
        val NUM_PREDICTIONS = OUTPUT_NUM_PREDICTIONS
        val NUM_PARAMS = OUTPUT_NUM_PARAMS
        val NUM_CLASSES = NUM_PARAMS - 4

        val detections = mutableListOf<Detection>()
        val boundingBoxes = mutableListOf<Rect2d>()
        val confidences = mutableListOf<Float>()
        val classIndexes = mutableListOf<Int>()

        for (i in 0 until NUM_PREDICTIONS) {
            var currentMaxClassScore = 0f

            val centerX = output[0][0][i]
            val centerY = output[0][1][i]
            val width = output[0][2][i]
            val height = output[0][3][i]

            var maxClassScore = 0f
            var classIndex = -1
            for (j in 0 until NUM_CLASSES) {
                val classScore = output[0][4 + j][i]
                if (classScore > currentMaxClassScore) {
                    currentMaxClassScore = classScore
                    classIndex = j
                }
            }
            maxClassScore = currentMaxClassScore

            val finalConfidence = maxClassScore

            if (finalConfidence >= YOLO_CONFIDENCE_THRESHOLD) {
                if (classIndex != -1) {
                    val scaleX = frameWidth.toDouble() / YOLO_INPUT_SIZE
                    val scaleY = frameHeight.toDouble() / YOLO_INPUT_SIZE

                    val scaledCenterX = centerX * scaleX
                    val scaledCenterY = centerY * scaleY
                    val scaledWidth = width * scaleX
                    val scaledHeight = height * scaleY

                    val x1 = max(0.0, (scaledCenterX - scaledWidth / 2.0))
                    val y1 = max(0.0, (scaledCenterY - scaledHeight / 2.0))
                    val x2 = min(frameWidth.toDouble() - 1, (scaledCenterX + scaledWidth / 2.0))
                    val y2 = min(frameHeight.toDouble() - 1, (scaledCenterY + scaledHeight / 2.0))

                    val rectWidth = max(0.0, x2 - x1)
                    val rectHeight = max(0.0, y2 - y1)

                    if (rectWidth > 0 && rectHeight > 0) {
                        boundingBoxes.add(Rect2d(x1, y1, rectWidth, rectHeight))
                        confidences.add(finalConfidence)
                        classIndexes.add(classIndex)
                    }
                }
            }
        }

        if (boundingBoxes.isEmpty()) {
            return emptyList()
        }

        val boxesMat = MatOfRect2d(*boundingBoxes.toTypedArray())
        val confidencesMat = MatOfFloat(*confidences.toFloatArray())
        val indicesMat = MatOfInt()

        try {
            Dnn.NMSBoxes(boxesMat, confidencesMat, YOLO_CONFIDENCE_THRESHOLD, YOLO_IOU_THRESHOLD, indicesMat)
        } catch (e: Exception) {
            Log.e(TAG, "NMSBoxes Error: ${e.message}. Check OpenCV version and DNN module.")
            boxesMat.release()
            confidencesMat.release()
            indicesMat.release()
            return emptyList()
        }

        val keepIndices = indicesMat.toArray()
        for (index in keepIndices) {
            if (index < 0 || index >= boundingBoxes.size) {
                continue
            }
            val rect = boundingBoxes[index]
            val classIdx = classIndexes[index]
            val className = if (classIdx >= 0 && classIdx < NUM_CLASSES) {
                if (classIdx < cocoLabels.size) {
                    cocoLabels[classIdx]
                } else {
                    "Unknown ($classIdx)"
                }
            } else {
                "Unknown ($classIdx)"
            }

            val detection = Detection(
                Rect(rect.x.toInt(), rect.y.toInt(), rect.width.toInt(), rect.height.toInt()),
                confidences[index],
                classIdx,
                className
            )
            detections.add(detection)
        }

        boxesMat.release()
        confidencesMat.release()
        indicesMat.release()

        return detections
    }

    fun close() {
        tfliteInterpreter?.close()
        tfliteInterpreter = null
        isInitialized.set(false)
        Log.d(TAG, "YoloDetector closed.")
    }
}

