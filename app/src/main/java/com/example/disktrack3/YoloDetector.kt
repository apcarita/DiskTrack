package com.example.disktrack3

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.opencv.core.Rect
import org.opencv.core.Rect2d
import org.opencv.dnn.Dnn
import org.opencv.core.MatOfRect2d
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.HashMap
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min

class YoloDetector(
    context: Context
) {

    private val TAG = "YoloDetector"
    private val TFLITE_MODEL_FILE = "yolo11n_float16_r416.tflite"
    private val YOLO_INPUT_SIZE = 416
    private val YOLO_CONFIDENCE_THRESHOLD = 0.25f
    private val YOLO_IOU_THRESHOLD = 0.45f

    private var tfliteInterpreter: Interpreter? = null
    private val isInitialized = AtomicBoolean(false)
    private var inputByteBuffer: ByteBuffer? = null
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
    private val OUTPUT_NUM_PARAMS = 84
    private val OUTPUT_NUM_PREDICTIONS = 3549
    private val PERSON_CLASS_INDEX: Int = cocoLabels.indexOf("person")

    init {
        Log.d(TAG, "YoloDetector init: Starting initialization (CPU focus)...")
        if (PERSON_CLASS_INDEX == -1) {
            Log.e(TAG, "Initialization Error: Could not find 'person' class in cocoLabels!")
        } else {
            Log.d(TAG, "'person' class index: $PERSON_CLASS_INDEX")
        }

        try {
            val assetManager = context.assets
            val modelBuffer = loadModelFile(assetManager, TFLITE_MODEL_FILE)

            val interpreterOptions = Interpreter.Options()
            val numThreads = 4
            interpreterOptions.setNumThreads(numThreads)
            Log.i(TAG, "Setting TFLite CPU threads to: $numThreads")

            tfliteInterpreter = Interpreter(modelBuffer, interpreterOptions)

            // --- Input Buffer Setup (FLOAT32, updated size) ---
            // Size = Batch * Height * Width * Channels * BytesPerChannel (4 for FLOAT32)
            val inputBufferSize = 1 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE * 3 * 4 // Use updated YOLO_INPUT_SIZE
            Log.d(TAG, "Allocating FLOAT32 input buffer size: $inputBufferSize bytes for ${YOLO_INPUT_SIZE}x${YOLO_INPUT_SIZE}") // Log updated size
            inputByteBuffer = ByteBuffer.allocateDirect(inputBufferSize)
            inputByteBuffer?.order(ByteOrder.nativeOrder())

            val outputTensor = tfliteInterpreter?.getOutputTensor(0)
            val reportedOutputShape = outputTensor?.shape() ?: intArrayOf(0, 0, 0)
            val outputDataType = outputTensor?.dataType() ?: DataType.FLOAT32
            Log.d(TAG, "TFLite Reported Output Shape: ${reportedOutputShape.joinToString()}, DataType: $outputDataType")

            if (reportedOutputShape.size != 3 || reportedOutputShape[0] != 1 || reportedOutputShape[1] != OUTPUT_NUM_PARAMS || reportedOutputShape[2] != OUTPUT_NUM_PREDICTIONS) {
                Log.e(TAG, "FATAL: Output tensor shape ${reportedOutputShape.joinToString()} does NOT match expected [1, $OUTPUT_NUM_PARAMS, $OUTPUT_NUM_PREDICTIONS]. Check model or constants.")
                isInitialized.set(false)
            } else if (outputDataType != DataType.FLOAT32) {
                Log.w(TAG, "Warning: Output tensor data type is $outputDataType, but processing assumes FLOAT32.")
                isInitialized.set(true)
            } else {
                isInitialized.set(true)
            }

            if (isInitialized.get()) {
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

                Log.d(TAG, "YoloDetector initialized successfully (CPU focus, FLOAT32 input, ${YOLO_INPUT_SIZE}x${YOLO_INPUT_SIZE}, ${OUTPUT_NUM_PREDICTIONS} predictions)")
            } else {
                tfliteInterpreter?.close()
                tfliteInterpreter = null
                Log.e(TAG, "YoloDetector initialization failed due to output shape mismatch.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing YoloDetector (CPU focus): ${e.message}", e)
            tfliteInterpreter?.close()
            tfliteInterpreter = null
            isInitialized.set(false)
        }
        Log.d(TAG, "YoloDetector init: Initialization block finished (CPU focus). Final Ready State: ${isInitialized.get()}")
    }

    fun isReady(): Boolean = isInitialized.get()

    @Throws(Exception::class)
    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        fileChannel.close()
        inputStream.close()
        fileDescriptor.close()
        return mappedByteBuffer
    }

    data class Detection(val boundingBox: Rect, val confidence: Float, val classIndex: Int, val className: String)

    fun detect(bitmap: Bitmap, frameWidth: Int, frameHeight: Int): List<Detection> {
        if (!isInitialized.get() || tfliteInterpreter == null || inputByteBuffer == null) {
            Log.w(TAG, "Detector not initialized or input buffer is null.")
            return emptyList()
        }

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, true)

        inputByteBuffer?.rewind()
        val intValues = IntArray(YOLO_INPUT_SIZE * YOLO_INPUT_SIZE)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        // --- Populate FLOAT32 ByteBuffer ---
        var pixel = 0
        for (i in 0 until YOLO_INPUT_SIZE) {
            for (j in 0 until YOLO_INPUT_SIZE) {
                val value = intValues[pixel++]
                val r = ((value shr 16 and 0xFF) / 255.0f)
                val g = ((value shr 8 and 0xFF) / 255.0f)
                val b = ((value and 0xFF) / 255.0f)
                inputByteBuffer?.putFloat(r)
                inputByteBuffer?.putFloat(g)
                inputByteBuffer?.putFloat(b)
            }
        }
        resizedBitmap.recycle()

        val outputs = HashMap<Int, Any>()
        val BATCH_SIZE = 1
        val outputArray = Array(BATCH_SIZE) { Array(OUTPUT_NUM_PARAMS) { FloatArray(OUTPUT_NUM_PREDICTIONS) } }
        outputs[0] = outputArray

        try {
            tfliteInterpreter?.runForMultipleInputsOutputs(arrayOf(inputByteBuffer), outputs)
        } catch (e: Exception) {
            Log.e(TAG, "TFLite inference error: ${e.message}", e)
            return emptyList()
        }

        val outputData = outputs[0] as? Array<Array<FloatArray>>
        if (outputData == null) {
            Log.e(TAG, "Output data retrieval failed or incorrect type.")
            return emptyList()
        }

        return processOutput(outputData, frameWidth, frameHeight)
    }

    private fun processOutput(output: Array<Array<FloatArray>>, frameWidth: Int, frameHeight: Int): List<Detection> {
        val NUM_PREDICTIONS = OUTPUT_NUM_PREDICTIONS
        val NUM_PARAMS = OUTPUT_NUM_PARAMS
        val NUM_CLASSES = NUM_PARAMS - 4

        val personBoxes = mutableListOf<Rect2d>()
        val personConfidences = mutableListOf<Float>()

        for (i in 0 until NUM_PREDICTIONS) {
            var maxClassScore = 0f
            var classIndex = -1
            for (j in 0 until NUM_CLASSES) {
                val classScore = output[0][4 + j][i]
                if (classScore > maxClassScore) {
                    maxClassScore = classScore
                    classIndex = j
                }
            }

            if (classIndex == PERSON_CLASS_INDEX && maxClassScore >= YOLO_CONFIDENCE_THRESHOLD) {
                val centerX = output[0][0][i]
                val centerY = output[0][1][i]
                val width = output[0][2][i]
                val height = output[0][3][i]

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
                    personBoxes.add(Rect2d(x1, y1, rectWidth, rectHeight))
                    personConfidences.add(maxClassScore)
                }
            }
        }

        if (personBoxes.isEmpty()) {
            return emptyList()
        }

        val boxesMat = MatOfRect2d(*personBoxes.toTypedArray())
        val confidencesMat = MatOfFloat(*personConfidences.toFloatArray())
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

        val finalDetections = mutableListOf<Detection>()
        val keepIndices = indicesMat.toArray()

        for (index in keepIndices) {
            if (index < 0 || index >= personBoxes.size) {
                Log.w(TAG, "NMS returned invalid index: $index (size: ${personBoxes.size})")
                continue
            }
            val rect = personBoxes[index]
            val confidence = personConfidences[index]
            val className = cocoLabels[PERSON_CLASS_INDEX]

            val detection = Detection(
                Rect(rect.x.toInt(), rect.y.toInt(), rect.width.toInt(), rect.height.toInt()),
                confidence,
                PERSON_CLASS_INDEX,
                className
            )
            finalDetections.add(detection)
        }

        boxesMat.release()
        confidencesMat.release()
        indicesMat.release()

        return finalDetections
    }

    fun close() {
        tfliteInterpreter?.close()
        tfliteInterpreter = null
        inputByteBuffer = null
        isInitialized.set(false)
        Log.d(TAG, "YoloDetector closed.")
    }
}

