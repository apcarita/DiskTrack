package com.example.disktrack

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

private const val TAG = "ImageAnalyzer"

class PlayerImageAnalyzer(
    private val onDebugImageProcessed: (Bitmap) -> Unit = {}
) : ImageAnalysis.Analyzer {

    private var frameCount = 0
    
    // Performance optimization
    private val PROCESSING_INTERVAL = 3 // Process every 3rd frame for better responsiveness
    
    // Add new variables for sideline and horizon detection
    private var framesSinceLastLineDetection = 0
    private var currentSidelineSlope = 0.0 // Slope of detected sideline
    private var currentSidelineIntercept = 0 // Y-intercept of detected sideline
    private var currentHorizonY = 0 // Y position of horizon line
    private val LINE_DETECTION_INTERVAL = 60 // Increased interval for line detection (every 60 frames)
    private val NUM_HORIZONTAL_SAMPLES = 8 // Number of horizontal sample points to check
    private val NUM_VERTICAL_SAMPLES = 5 // Number of vertical samples for horizon detection
    
    @androidx.camera.core.ExperimentalGetImage
    override fun analyze(imageProxy: ImageProxy) {
        val startTime = System.currentTimeMillis()
        
        try {
            // Process only every Nth frame to improve performance
            if (frameCount++ % PROCESSING_INTERVAL != 0) {
                imageProxy.close()
                return
            }
            
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
            Log.d(TAG, "Downsampled bitmap: ${downsampledBitmap.width}x${downsampledBitmap.height}")
            
            // Create a binary debug bitmap with sideline detection
            val debugBitmap = createFieldVisualizationWithSideline(downsampledBitmap)
            
            // Pass the debug bitmap to the callback - this is our main display now
            Log.d(TAG, "Sending visualization bitmap: ${debugBitmap.width}x${debugBitmap.height}")
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
    
    private fun createFieldVisualizationWithSideline(original: Bitmap): Bitmap {
        val width = original.width
        val height = original.height
        
        // Create a new bitmap 
        val debugBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        
        // Detect sideline and horizon periodically to reduce computational cost
        if (framesSinceLastLineDetection >= LINE_DETECTION_INTERVAL) {
            detectSideline(original)
            detectHorizon(original)
            framesSinceLastLineDetection = 0
        } else {
            framesSinceLastLineDetection++
        }
        
        // Get horizon position (default to 50% of height if not detected)
        val horizonY = if (currentHorizonY > 0) currentHorizonY else height / 2
        
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
            }
        }
        
        // Rotate the bitmap 90 degrees to be upright when phone is horizontal
        return rotateBitmap(debugBitmap, 90f)
    }
    
    private fun detectSideline(bitmap: Bitmap) {
        val width = bitmap.width
        val height = bitmap.height
        val halfHeight = height / 2
        
        // This will hold the top-most non-green pixels (potential players) at sample points
        val samplePoints = mutableListOf<Pair<Int, Int>>()
        
        // Sample at evenly spaced horizontal intervals
        val sampleSpacing = width / (NUM_HORIZONTAL_SAMPLES + 1)
        
        // For each sample point, scan from the middle down to find first significant non-green blob
        for (sampleIndex in 1..NUM_HORIZONTAL_SAMPLES) {
            val x = sampleIndex * sampleSpacing
            
            // Start at the middle of the image and scan toward the bottom
            // Looking for clusters of non-green pixels (potential players)
            var blobFound = false
            var consecutiveNonGreen = 0
            var topOfBlob = halfHeight
            
            for (y in halfHeight until height) {
                val pixel = bitmap.getPixel(x, y)
                
                if (!isGreen(pixel)) {
                    consecutiveNonGreen++
                    
                    // Consider a significant blob when we find several consecutive non-green pixels
                    if (consecutiveNonGreen > 5 && !blobFound) {
                        topOfBlob = y - 5 // Adjust to get the top of the blob
                        blobFound = true
                        break
                    }
                } else {
                    consecutiveNonGreen = 0
                }
            }
            
            if (blobFound) {
                samplePoints.add(Pair(x, topOfBlob))
            }
        }
        
        // If we found enough points, fit a line using linear regression
        if (samplePoints.size >= 3) {
            // Simple linear regression to find the best-fit line
            // y = mx + b
            
            // Calculate the means
            val n = samplePoints.size
            val sumX = samplePoints.sumOf { it.first.toDouble() }
            val sumY = samplePoints.sumOf { it.second.toDouble() }
            val meanX = sumX / n
            val meanY = sumY / n
            
            // Calculate slope (m) using a simplified least squares method
            var numerator = 0.0
            var denominator = 0.0
            
            for (point in samplePoints) {
                val xDiff = point.first - meanX
                val yDiff = point.second - meanY
                numerator += xDiff * yDiff
                denominator += xDiff * xDiff
            }
            
            // Avoid division by zero
            if (denominator != 0.0) {
                val slope = numerator / denominator
                val intercept = meanY - slope * meanX
                
                // Update the current sideline equation
                currentSidelineSlope = slope
                currentSidelineIntercept = intercept.toInt()
                
                Log.d(TAG, "Detected sideline: y = ${currentSidelineSlope}x + $currentSidelineIntercept")
            }
        }
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
        
        // Constrain to reasonable values (between 1/2 and 3/4 of height)
        return y.coerceIn(height / 2, (height * 3) / 4)
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
    
    // Add a function to rotate the bitmap
    private fun rotateBitmap(source: Bitmap, degrees: Float): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height, matrix, true
        )
    }
}
