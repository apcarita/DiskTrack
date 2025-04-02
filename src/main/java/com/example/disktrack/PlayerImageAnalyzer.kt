package com.example.disktrack

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import kotlin.math.abs  // Add this import for the abs function
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

private const val TAG = "ImageAnalyzer"

class PlayerImageAnalyzer(
    private val onDebugImageProcessed: (Bitmap) -> Unit = {}
) : ImageAnalysis.Analyzer {

    private var frameCount = 0
    
    // Line detection variables
    private var currentSidelineSlope = 0.0 // Slope of detected sideline
    private var currentSidelineIntercept = 0 // Y-intercept of detected sideline
    private var currentHorizonY = 0 // Y position of horizon line
    
    // Detection intervals - using lower numbers for more frequent updates
    private val SIDELINE_DETECTION_INTERVAL = 10 // Detect sideline every 10 frames
    private val HORIZON_DETECTION_INTERVAL = 10 // Detect horizon every 10 frames
    private val HORIZON_DETECTION_OFFSET = 5 // Offset detection to avoid running both at once
    
    // Updated sample count for sideline detection
    private val NUM_HORIZONTAL_SAMPLES = 30 // Reduced to 30 scan lines
    private val NUM_VERTICAL_SAMPLES = 5 // Sample points for horizon
    
    @androidx.camera.core.ExperimentalGetImage
    override fun analyze(imageProxy: ImageProxy) {
        val startTime = System.currentTimeMillis()
        
        try {
            // Process every frame
            frameCount++
            
            Log.d(TAG, "Analyzing frame #$frameCount")
            val image = imageProxy.image
            if (image == null) {
                Log.e(TAG, "Image is null")
                imageProxy.close()
                return
            }
            
            val bitmap = imageProxy.toBitmap()
            if (bitmap == null) {
                Log.e(TAG, "Failed to convert image to bitmap")
                imageProxy.close()
                return
            }
            
            // Downsample the bitmap by 2x for better performance but still good detail
            val downsampledBitmap = downsampleBitmap(bitmap, 2)
            
            // Decide if we need to detect lines in this frame
            val detectSideline = frameCount % SIDELINE_DETECTION_INTERVAL == 0
            val detectHorizon = frameCount % HORIZON_DETECTION_INTERVAL == HORIZON_DETECTION_OFFSET
            
            // Create a binary debug bitmap with sideline and horizon
            val debugBitmap = createFieldVisualization(
                downsampledBitmap, 
                detectSideline, 
                detectHorizon
            )
            
            // Pass the debug bitmap to the callback - this is our main display now
            onDebugImageProcessed(debugBitmap)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error analyzing image", e)
        } finally {
            val processingTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "Frame analysis took $processingTime ms")
            imageProxy.close()
        }
    }
    
    private fun downsampleBitmap(original: Bitmap, factor: Int): Bitmap {
        return Bitmap.createScaledBitmap(
            original,
            original.width / factor,
            original.height / factor,
            true
        )
    }
    
    private fun isGreen(pixel: Int): Boolean {
        // Extract RGB components
        val r = Color.red(pixel)
        val g = Color.green(pixel)
        val b = Color.blue(pixel)
        
        // Permissive green detection to include darker greens
        return (g > 45) && 
               ((g > r - 40) || (g > b - 15)) && 
               (g * 0.7 > b * 0.8)
    }
    
    private fun createFieldVisualization(
        original: Bitmap, 
        detectSideline: Boolean, 
        detectHorizon: Boolean
    ): Bitmap {
        val width = original.width
        val height = original.height
        
        // Create a new bitmap 
        val debugBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        
        // Use a consistent order - detect horizon first, then sideline that starts from horizon
        if (detectHorizon) {
            detectHorizon(original)
            Log.d(TAG, "Horizon detection ran on frame $frameCount")
        }
        
        // Store sample points for visualization
        val sampleColumns = mutableListOf<Int>()
        val sampleEndPoints = mutableListOf<Pair<Int, Int>>()
        
        // Detect sideline after horizon, so it can use the horizon line
        if (detectSideline) {
            val sidelineResult = detectSideline(original)
            sampleColumns.addAll(sidelineResult.first)
            sampleEndPoints.addAll(sidelineResult.second)
            Log.d(TAG, "Sideline detection ran on frame $frameCount")
        }
        
        // Get horizon position (default to 1/3 of height if not detected)
        val horizonY = if (currentHorizonY > 0) currentHorizonY else height / 3
        
        for (x in 0 until width) {
            // Calculate the y-position of the sideline at this x-position
            val sidelineY = calculateSidelineY(x, width, height)
            
            for (y in 0 until height) {
                val pixel = original.getPixel(x, y)
                
                // Determine if pixel is in the playable field area
                // - Below horizon
                // - Above sideline
                val belowHorizon = y > horizonY
                val aboveSideline = y < sidelineY
                val inPlayableArea = belowHorizon && aboveSideline
                
                if (isGreen(pixel)) {
                    if (inPlayableArea) {
                        // White for green pixels in playable field area
                        debugBitmap.setPixel(x, y, Color.WHITE)
                    } else {
                        // Light gray for green pixels outside playable area
                        debugBitmap.setPixel(x, y, Color.rgb(180, 180, 180))
                    }
                } else {
                    if (inPlayableArea) {
                        // Black for non-green pixels in playable field area
                        debugBitmap.setPixel(x, y, Color.BLACK)
                    } else {
                        // Dark gray for non-green pixels outside playable area
                        debugBitmap.setPixel(x, y, Color.rgb(60, 60, 60))
                    }
                }
                
                // Draw the sideline for visualization
                if (Math.abs(y - sidelineY) < 2) {
                    debugBitmap.setPixel(x, y, Color.RED)
                }
                
                // Draw the horizon line for visualization
                if (Math.abs(y - horizonY) < 2) {
                    debugBitmap.setPixel(x, y, Color.BLUE)
                }
                
                // Draw vertical lines at each sample column showing scan direction (bottom to top)
                if (x in sampleColumns) {
                    // Find the endpoint for this column
                    val endPoint = sampleEndPoints.find { it.first == x }?.second ?: (height * 2 / 3)
                    
                    // Draw yellow vertical line from bottom up to the detected sideline point
                    if (y >= endPoint - 15 && y <= endPoint) {
                        // Color the exact sideline point red for visibility
                        if (y == endPoint) {
                            debugBitmap.setPixel(x, y, Color.RED)
                        } else {
                            // Color the 15 pixels above it yellow to show the check area
                            debugBitmap.setPixel(x, y, Color.YELLOW) 
                        }
                    }
                }
            }
        }
        
        // Rotate the bitmap 90 degrees to be upright when phone is horizontal
        return rotateBitmap(debugBitmap, 90f)
    }
    
    private fun detectSideline(bitmap: Bitmap): Pair<List<Int>, List<Pair<Int, Int>>> {
        val width = bitmap.width
        val height = bitmap.height
        
        // This will hold the sideline points (where we transition from field to non-field)
        val samplePoints = mutableListOf<Pair<Int, Int>>()
        
        // Sample at evenly spaced horizontal intervals
        val sampleSpacing = width / (NUM_HORIZONTAL_SAMPLES + 1)
        
        // Store columns and endpoints for visualization
        val sampleColumns = mutableListOf<Int>()
        val sampleEndpoints = mutableListOf<Pair<Int, Int>>()
        
        // For each sample column, scan from bottom up to find the sideline
        for (sampleIndex in 1..NUM_HORIZONTAL_SAMPLES) {
            val x = sampleIndex * sampleSpacing
            sampleColumns.add(x) // Add this column for visualization
            
            var sidelineY = height * 2 / 3 // Default position if no sideline found
            var foundSideline = false
            
            // Scan from bottom up to find the last black pixel with 15 consecutive white pixels above it
            for (y in height - 1 downTo height / 4) {
                val pixel = bitmap.getPixel(x, y)
                
                // Check if current pixel is non-green (black in our visualization)
                if (!isGreen(pixel)) {
                    // Check if there are 15 consecutive green (white) pixels above this point
                    var consecutiveGreenCount = 0
                    var allGreen = true
                    
                    // Look at the 15 pixels above this point
                    for (checkY in y - 1 downTo y - 15) {
                        // Make sure we don't go off the top of the image
                        if (checkY < 0) {
                            allGreen = false
                            break
                        }
                        
                        // Check if pixel is green
                        if (isGreen(bitmap.getPixel(x, checkY))) {
                            consecutiveGreenCount++
                        } else {
                            allGreen = false
                            break
                        }
                    }
                    
                    // If we found 15 consecutive green pixels above this black pixel, we found the sideline
                    if (consecutiveGreenCount >= 15 && allGreen) {
                        sidelineY = y
                        foundSideline = true
                        break
                    }
                }
            }
            
            // Add the found sideline point to our collections
            if (foundSideline) {
                samplePoints.add(Pair(x, sidelineY))
            }
            
            // Always add an endpoint for visualization
            sampleEndpoints.add(Pair(x, sidelineY))
        }
        
        // If we found enough points, fit a line using linear regression
        if (samplePoints.size >= 3) {
            // Remove any outliers to improve line fit
            val yValues = samplePoints.map { it.second }
            val median = calculateMedian(yValues)
            val mad = calculateMAD(yValues, median)
            
            // Filter out points that are too far from the median
            val filteredPoints = samplePoints.filter { 
                abs(it.second - median) < 2.5 * mad 
            }
            
            if (filteredPoints.size >= 3) {
                // Simple linear regression to find the best-fit line (y = mx + b)
                // Calculate the means
                val n = filteredPoints.size
                val sumX = filteredPoints.sumOf { it.first.toDouble() }
                val sumY = filteredPoints.sumOf { it.second.toDouble() }
                val meanX = sumX / n
                val meanY = sumY / n
                
                // Calculate slope (m) using a simplified least squares method
                var numerator = 0.0
                var denominator = 0.0
                
                for (point in filteredPoints) {
                    val xDiff = point.first - meanX
                    val yDiff = point.second - meanY
                    numerator += xDiff * yDiff
                    denominator += xDiff * xDiff
                }
                
                // Avoid division by zero
                if (denominator != 0.0) {
                    val slope = numerator / denominator
                    
                    // Limit the slope to a reasonable range to prevent extreme angles
                    val limitedSlope = slope.coerceIn(-0.5, 0.5)
                    
                    val intercept = meanY - limitedSlope * meanX
                    
                    // Update the current sideline equation
                    currentSidelineSlope = limitedSlope
                    currentSidelineIntercept = intercept.toInt()
                    
                    Log.d(TAG, "Detected sideline: y = ${currentSidelineSlope}x + $currentSidelineIntercept with ${filteredPoints.size} points")
                }
            }
        }
        
        // Return the columns and endpoints for visualization
        return Pair(sampleColumns, sampleEndpoints)
    }
    
    private fun detectHorizon(bitmap: Bitmap) {
        val width = bitmap.width
        val height = bitmap.height
        val quarterHeight = height / 4
        
        // This will hold sky-to-field transition points
        val horizonPoints = mutableListOf<Int>()
        
        // Sample at evenly spaced horizontal intervals
        val sampleSpacing = width / (NUM_VERTICAL_SAMPLES + 1)
        
        // For each sample column, scan from top to find the transition from sky to field
        for (sampleIndex in 1..NUM_VERTICAL_SAMPLES) {
            val x = sampleIndex * sampleSpacing
            
            // Scan from 10% to 75% height to find transition from sky to field
            var skyToFieldTransition = quarterHeight  // Default to 25% of height
            var consecutiveGreen = 0
            
            for (y in (height / 10) until (height * 3 / 4)) {
                val pixel = bitmap.getPixel(x, y)
                
                if (isGreen(pixel)) {
                    consecutiveGreen++
                    
                    // Consider a field transition when we find several consecutive green pixels
                    if (consecutiveGreen > 5) {
                        skyToFieldTransition = y - 5  // Adjust to get the start of green
                        break
                    }
                } else {
                    consecutiveGreen = 0
                }
            }
            
            horizonPoints.add(skyToFieldTransition)
        }
        
        // If we found samples, calculate the average horizon position
        if (horizonPoints.isNotEmpty()) {
            // Sort and remove outliers
            horizonPoints.sort()
            val validPoints = if (horizonPoints.size > 3) {
                // Discard the highest and lowest values to remove outliers
                horizonPoints.subList(1, horizonPoints.size - 1)
            } else {
                horizonPoints
            }
            
            // Calculate average horizon position
            val averageHorizon = validPoints.average().toInt()
            
            // Update the current horizon position
            currentHorizonY = averageHorizon
            Log.d(TAG, "Detected horizon at y = $currentHorizonY")
        }
    }
    
    private fun calculateSidelineY(x: Int, width: Int, height: Int): Int {
        // Default: horizontal line at 2/3 of the height if no sideline detected
        if (currentSidelineSlope == 0.0 && currentSidelineIntercept == 0) {
            return (height * 2) / 3
        }
        
        // Calculate y position using the line equation: y = mx + b
        val y = (currentSidelineSlope * x + currentSidelineIntercept).toInt()
        
        // Constrain to reasonable values (between 1/3 and 3/4 of height)
        // Expanded the range to allow for more angled sidelines
        return y.coerceIn(height / 3, (height * 3) / 4)
    }
    
    // Helper function to calculate median
    private fun calculateMedian(values: List<Number>): Double {
        val sortedValues = values.map { it.toDouble() }.sorted()
        val mid = sortedValues.size / 2
        return if (sortedValues.size % 2 == 0) {
            (sortedValues[mid - 1] + sortedValues[mid]) / 2.0
        } else {
            sortedValues[mid]
        }
    }

    // Helper function to calculate Median Absolute Deviation (robust measure of variability)
    private fun calculateMAD(values: List<Int>, median: Double): Double {
        val deviations = values.map { abs(it - median) }
        return calculateMedian(deviations)
    }
    
    @androidx.camera.core.ExperimentalGetImage
    private fun ImageProxy.toBitmap(): Bitmap? {
        val image = this.image ?: return null
        
        val planes = image.planes
        val buffer = planes[0].buffer
        val pixelStride = planes[0].pixelStride
        val rowStride = planes[0].rowStride
        val rowPadding = rowStride - pixelStride * width
        
        // Create bitmap
        val bitmap = Bitmap.createBitmap(
            width + rowPadding / pixelStride,
            height,
            Bitmap.Config.ARGB_8888
        )
        
        bitmap.copyPixelsFromBuffer(buffer)
        return bitmap
    }
    
    private fun rotateBitmap(source: Bitmap, degrees: Float): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height, matrix, true
        )
    }
}
