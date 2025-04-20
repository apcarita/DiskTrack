package com.example.disktrack3

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType // Import CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point // Import Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import org.opencv.core.Size // Import Size for GaussianBlur

class CannyEdgeProcessor {

    private val TAG = "CannyEdgeProcessor"

    // Canny thresholds
    private val lowerThreshold = 100.0
    private val upperThreshold = 300.0

    // Color for drawing bounding boxes
    private val boxColor = Scalar(255.0, 0.0, 0.0) // Red (Stationary)
    private val movingBoxColor = Scalar(0.0, 0.0, 255.0) // Blue (Moving)
    private val horizonColor = Scalar(0.0, 255.0, 0.0) // Green for Horizon Line

    // --- Contour Filtering Parameters (Post-Merge - Currently Disabled) ---
    private val MAX_CONTOUR_WIDTH_POST_MERGE = 50.0 // Max width in pixels
    private val MIN_ASPECT_RATIO_POST_MERGE = 0.25 // Width/Height (e.g., 1:4)
    private val MAX_ASPECT_RATIO_POST_MERGE = 0.5  // Width/Height (e.g., 1:2)
    // --- End Post-Merge Contour Filtering Parameters ---

    // --- Pre-Merge Contour Filtering Parameters (Remove Flat Boxes) ---
    private val MIN_CONTOUR_HEIGHT_PRE_MERGE = 5.0 // Minimum height in pixels
    private val MAX_ASPECT_RATIO_PRE_MERGE = 5.0 // Maximum Width/Height ratio (e.g., 5:1)
    // --- End Pre-Merge Contour Filtering Parameters ---

    // --- Final Draw Filtering Parameters ---
    private val MIN_HEIGHT_TO_WIDTH_RATIO = 2.0 // Minimum Height/Width ratio for drawing
    // --- End Final Draw Filtering Parameters ---

    // --- Horizon Finding Parameters ---
    private val HORIZON_BOX_THRESHOLD = 18 // Min number of boxes horizon must intersect
    // --- End Horizon Finding Parameters ---

    // --- Contour Merging Parameters ---
    private val MERGE_THRESHOLD = 0.5 // Max distance in pixels between boxes to merge - NO LONGER USED FOR MERGING CONDITION
    private val MERGE_IOU_THRESHOLD = 0.3 // Minimum IoU for merging boxes
    // --- End Contour Merging Parameters ---

    // --- Motion Detection ---
    private var previousBoxes: List<Rect> = emptyList()
    private val IOU_STATIONARY_THRESHOLD = 0.3 // Min IoU to consider a box stationary
    // --- End Motion Detection ---

    // Helper function to check if two rectangles are close enough to merge
    private fun areBoxesClose(rect1: Rect, rect2: Rect, threshold: Double): Boolean {
        val expandedRect1 = Rect(
            (rect1.x - threshold).toInt(),
            (rect1.y - threshold).toInt(),
            (rect1.width + 2 * threshold).toInt(),
            (rect1.height + 2 * threshold).toInt()
        )
        // Check for overlap between expanded rect1 and rect2
        return expandedRect1.x < rect2.x + rect2.width &&
               expandedRect1.x + expandedRect1.width > rect2.x &&
               expandedRect1.y < rect2.y + rect2.height &&
               expandedRect1.y + expandedRect1.height > rect2.y
    }

    // Helper function to calculate the union of two rectangles
    private fun unionRects(rect1: Rect, rect2: Rect): Rect {
        val x = min(rect1.x, rect2.x)
        val y = min(rect1.y, rect2.y)
        val width = max(rect1.x + rect1.width, rect2.x + rect2.width) - x
        val height = max(rect1.y + rect1.height, rect2.y + rect2.height) - y
        return Rect(x, y, width, height)
    }

    // Helper function to calculate Intersection over Union (IoU)
    private fun calculateIoU(rect1: Rect, rect2: Rect): Double {
        val xA = max(rect1.x, rect2.x)
        val yA = max(rect1.y, rect2.y)
        val xB = min(rect1.x + rect1.width, rect2.x + rect2.width)
        val yB = min(rect1.y + rect1.height, rect2.y + rect2.height)

        // Calculate intersection area
        val intersectionArea = max(0, xB - xA) * max(0, yB - yA).toDouble()

        // Calculate union area
        val box1Area = rect1.width * rect1.height.toDouble()
        val box2Area = rect2.width * rect2.height.toDouble()
        val unionArea = box1Area + box2Area - intersectionArea

        // Compute IoU
        return if (unionArea > 0) intersectionArea / unionArea else 0.0
    }

    fun processFrame(inputFrame: Mat): Mat {
        val channels = mutableListOf<Mat>()
        val combinedRGB = Mat()
        val binaryEdges = Mat()
        val rgbaEdges = Mat()

        try {
            // Split channels, combine RGB, apply Canny
            Core.split(inputFrame, channels)
            if (channels.size < 3) {
                Log.e(TAG, "Error: Input frame does not have at least 3 channels (R, G, B).")
                return inputFrame
            }
            val redChannel = channels[0]
            val greenChannel = channels[1]
            val blueChannel = channels[2]

            val tempRG = Mat()
            Core.addWeighted(redChannel, 0.5, greenChannel, 0.5, 0.0, tempRG)
            Core.addWeighted(tempRG, 2.0 / 3.0, blueChannel, 1.0 / 3.0, 0.0, combinedRGB)
            tempRG.release()

            // --- Add Gaussian Blur for Noise Reduction ---
            Imgproc.GaussianBlur(combinedRGB, combinedRGB, Size(5.0, 5.0), 0.0)
            // --- End Gaussian Blur ---

            // Apply Canny on the combined RGB image directly with adjusted thresholds
            Imgproc.Canny(combinedRGB, binaryEdges, lowerThreshold, upperThreshold)

            // Find Contours
            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(
                binaryEdges, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
            )
            hierarchy.release()

            // Convert to RGBA for Drawing
            Imgproc.cvtColor(binaryEdges, rgbaEdges, Imgproc.COLOR_GRAY2RGBA)

            // Get frame dimensions for filtering and horizon
            val frameHeight = rgbaEdges.height()
            val frameWidth = rgbaEdges.width()
            val MAX_BOX_HEIGHT = frameHeight / 5.0

            // Get initial bounding boxes
            var currentBoxes = contours.map { Imgproc.boundingRect(it) }.toMutableList()
            contours.forEach { it.release() } // Release original contours
            Log.d(TAG, "Initial contours found: ${currentBoxes.size}")

            // --- Pre-Filter Flat Contours ---
            val initialBoxCount = currentBoxes.size
            currentBoxes = currentBoxes.filter { box ->
                val height = box.height.toDouble()
                val width = box.width.toDouble()
                val aspectRatio = if (height > 0) width / height else Double.MAX_VALUE // Treat zero height as infinite aspect ratio

                val heightOk = height >= MIN_CONTOUR_HEIGHT_PRE_MERGE
                val ratioOk = aspectRatio <= MAX_ASPECT_RATIO_PRE_MERGE

                heightOk && ratioOk
            }.toMutableList()
            val filteredCount = initialBoxCount - currentBoxes.size
            Log.d(TAG, "Pre-filtered $filteredCount flat boxes. Remaining: ${currentBoxes.size}")
            // --- End Pre-Filter Flat Contours ---

            // --- Contour Merging Logic ---
            var mergedInLastPass: Boolean
            do {
                mergedInLastPass = false
                val nextBoxes = mutableListOf<Rect>()
                val mergedIndices = BooleanArray(currentBoxes.size) { false }

                for (i in currentBoxes.indices) {
                    if (mergedIndices[i]) continue

                    var currentMergedBox = currentBoxes[i]

                    for (j in (i + 1) until currentBoxes.size) {
                        if (mergedIndices[j]) continue

                        // --- Merge Condition Changed to IoU ---
                        if (calculateIoU(currentMergedBox, currentBoxes[j]) >= MERGE_IOU_THRESHOLD) {
                            currentMergedBox = unionRects(currentMergedBox, currentBoxes[j])
                            mergedIndices[j] = true
                            mergedInLastPass = true
                        }
                    }
                    nextBoxes.add(currentMergedBox)
                }

                currentBoxes = nextBoxes // Update list for the next pass

            } while (mergedInLastPass)
            Log.d(TAG, "IoU Merging complete. Final box count: ${currentBoxes.size}")
            // --- End Contour Merging Logic ---

            // --- Post-Merge Height Filter ---
            val countBeforeHeightFilter = currentBoxes.size
            currentBoxes = currentBoxes.filter { box ->
                box.height <= MAX_BOX_HEIGHT
            }.toMutableList()
            val filteredByHeightCount = countBeforeHeightFilter - currentBoxes.size
            Log.d(TAG, "Filtered $filteredByHeightCount boxes exceeding max height (1/5 screen). Remaining: ${currentBoxes.size}")
            // --- End Post-Merge Height Filter ---

            // --- Determine Motion Status ---
            val motionStatus = BooleanArray(currentBoxes.size) { true } // Default to moving
            if (previousBoxes.isNotEmpty()) {
                for (i in currentBoxes.indices) {
                    var maxIoU = 0.0
                    for (prevBox in previousBoxes) {
                        maxIoU = max(maxIoU, calculateIoU(currentBoxes[i], prevBox))
                    }
                    if (maxIoU >= IOU_STATIONARY_THRESHOLD) {
                        motionStatus[i] = false // Mark as stationary
                    }
                }
            }
            // --- End Determine Motion Status ---

            // --- Identify Candidate Boxes for Horizon Finding ---
            val candidateBoxesForHorizon = currentBoxes.filterIndexed { index, box ->
                val isStationary = !motionStatus[index]
                val ratioOk = box.height >= MIN_HEIGHT_TO_WIDTH_RATIO * box.width
                isStationary && ratioOk
            }
            Log.d(TAG, "Found ${candidateBoxesForHorizon.size} candidate boxes for horizon check.")
            // --- End Identify Candidate Boxes ---

            // --- Find Horizon Line ---
            var horizonY = 0 // Default to top of screen
            for (y in frameHeight / 2 downTo 1) {
                var intersectingCount = 0
                for (box in candidateBoxesForHorizon) {
                    if (box.y < y && box.y + box.height >= y) { // Check if line y intersects the box
                        intersectingCount++
                    }
                }
                if (intersectingCount >= HORIZON_BOX_THRESHOLD) {
                    horizonY = y
                    Log.d(TAG, "Horizon found at Y=$horizonY with $intersectingCount intersecting boxes.")
                    break // Found the horizon
                }
            }
            if (horizonY == 0) {
                Log.d(TAG, "Horizon threshold not met, defaulting horizon to top (Y=0).")
            }
            // --- End Find Horizon Line ---

            // Filter and Draw FINAL Boxes based on motion, ratio, and horizon
            var contoursDrawn = 0
            var movingCount = 0
            var stationaryCount = 0
            var ratioFilteredCount = 0
            var horizonFilteredCount = 0 // Counter for horizon filter

            for (i in currentBoxes.indices) {
                val boundingRect = currentBoxes[i]
                val isMoving = motionStatus[i]

                if (!isMoving) {
                    val heightWidthRatioOk = boundingRect.height >= MIN_HEIGHT_TO_WIDTH_RATIO * boundingRect.width
                    if (heightWidthRatioOk) {
                        // --- Horizon Filter ---
                        if (boundingRect.y >= horizonY) {
                            Imgproc.rectangle(rgbaEdges, boundingRect.tl(), boundingRect.br(), boxColor, 2)
                            contoursDrawn++
                            stationaryCount++
                        } else {
                            horizonFilteredCount++ // Count boxes filtered by horizon
                        }
                        // --- End Horizon Filter ---
                    } else {
                        ratioFilteredCount++ // Count boxes filtered by ratio
                    }
                } else {
                    movingCount++ // Still count moving ones for logging
                }
            }

            // --- Draw Horizon Line ---
            if (horizonY > 0 && horizonY < frameHeight) { // Draw only if found and not exactly at the top/bottom edge
                 Imgproc.line(rgbaEdges, Point(0.0, horizonY.toDouble()), Point(frameWidth.toDouble(), horizonY.toDouble()), horizonColor, 2)
            }
            // --- End Draw Horizon Line ---

            // Update log message to include horizon filtering
            val totalBoxesBeforeDrawFilter = movingCount + stationaryCount + ratioFilteredCount + horizonFilteredCount
            Log.d(TAG, "Drew: $contoursDrawn stationary boxes below horizon (Total before draw filters: $totalBoxesBeforeDrawFilter, $movingCount moving ignored, $ratioFilteredCount ratio-filtered, $horizonFilteredCount horizon-filtered), Post-merge Size/Ratio Filtering Disabled") // Updated Log

            // --- Update Previous Boxes for next frame ---
            previousBoxes = currentBoxes.toList() // Create a copy

            return rgbaEdges

        } catch (e: Exception) {
            Log.e(TAG, "Error during Canny processing: ${e.message}", e)
            previousBoxes = emptyList() // Reset previous boxes on error
            return inputFrame
        } finally {
            channels.forEach { it.release() }
            combinedRGB.release()
            binaryEdges.release()
        }
    }
}