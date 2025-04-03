package com.example.disktrack

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import java.util.concurrent.Executors

private const val TAG = "MLImageAnalyzer"

class MLImageAnalyzer(
    private val onProcessedImageReady: (Bitmap) -> Unit
) : ImageAnalysis.Analyzer {
    
    // Create ML Kit image labeler
    private val labeler = ImageLabeling.getClient(
        ImageLabelerOptions.Builder()
            .setConfidenceThreshold(0.7f)
            .build()
    )
    
    // Create a background executor for image processing
    private val backgroundExecutor = Executors.newSingleThreadExecutor()
    
    private var frameCount = 0
    private var fieldThreshold = 0.4f  // Reduced from 0.5f for less aggressive filtering
    
    // Default sideline position (2/3 down the image)
    private var sidelineY = 0
    
    // Add player tracking variables
    private var playerBlobs = mutableListOf<Blob>()
    private var previousPlayerBlobs = mutableListOf<Blob>()
    private var movementDirection = MovementDirection.NONE
    private var directionConfidence = 0
    private val DIRECTION_CONFIDENCE_THRESHOLD = 5
    private val MIN_BLOB_SIZE = 50 // Minimum pixel count to be considered a player
    private val MAX_BLOB_SIZE = 2000 // Maximum pixel count to avoid detecting large areas
    
    // Define RGB thresholds for field detection - with further expanded ranges
    private val R_MIN = 30   // Reduced from 40
    private val R_MAX = 200  // Increased from 180
    private val G_MIN = 60   // Reduced from 70
    private val G_MAX = 220  // Increased from 200
    private val B_MIN = 15   // Reduced from 20
    private val B_MAX = 170  // Increased from 150
    private val MIN_G_MINUS_R_DIFF = 1   // Reduced from 3 - much more permissive
    private val MIN_G_MINUS_B_DIFF = 10  // Reduced from 15
    
    // Data class to represent player blobs
    data class Blob(
        val x: Int, // Center x position
        val y: Int, // Center y position
        val size: Int, // Number of pixels
        val id: Int = -1 // For tracking across frames
    )
    
    enum class MovementDirection {
        LEFT, RIGHT, NONE
    }
    
    @androidx.camera.core.ExperimentalGetImage
    override fun analyze(imageProxy: ImageProxy) {
        frameCount++
        
        // Process fewer frames for better performance
        if (frameCount % 3 != 0) {
            imageProxy.close()
            return
        }
        
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }
        
        // Force field detection to be true instead of using ML Kit for now
        // The ML Kit recognition is inconsistent with sports fields
        val forcedFieldDetection = true
        
        backgroundExecutor.execute {
            try {
                val bitmap = toBitmap(imageProxy)
                
                // Create and send visualization without waiting for ML Kit
                val visualizedBitmap = createSimpleVisualization(
                    bitmap,
                    forcedFieldDetection // Always assume field is present
                )
                
                // Send the processed image back
                onProcessedImageReady(visualizedBitmap)
            } catch (e: Exception) {
                Log.e(TAG, "Error processing image: ${e.message}")
            } finally {
                imageProxy.close()
            }
        }
    }
    
    private fun createSimpleVisualization(original: Bitmap?, isFieldPresent: Boolean): Bitmap {
        if (original == null) {
            // Create a blank bitmap if original is null
            return Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.BLACK)
            }
        }
        
        // Create a copy that we can modify - fix the type mismatch by using a non-null config
        val config = original.config ?: Bitmap.Config.ARGB_8888
        val result = original.copy(config, true)
        val width = result.width
        val height = result.height
        
        // Initialize sideline position if not set
        if (sidelineY == 0) {
            sidelineY = (height * 2) / 3
        }
        
        // Detect and track players in the playable area
        val newPlayerBlobs = mutableListOf<Blob>()
        
        if (isFieldPresent) {
            // Step 1: Binary segmentation for efficient blob detection
            val binaryImage = Array(height) { BooleanArray(width) }
            
            // Fill binary image with pixels in the playable area that could be players
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val pixel = original.getPixel(x, y)
                    
                    // Extract RGB components
                    val r = Color.red(pixel)
                    val g = Color.green(pixel)
                    val b = Color.blue(pixel)
                    
                    // Less aggressive green detection - more inclusive
                    val isGreen = isFieldGreen(r, g, b)
                    val isBelowHorizon = y > height / 3
                    val isAboveSideline = y < sidelineY
                    
                    // Mark as potential player if not green and in playable area
                    binaryImage[y][x] = !isGreen && isBelowHorizon && isAboveSideline
                }
            }
            
            // Step 2: Simple connected components algorithm to find blobs
            val visited = Array(height) { BooleanArray(width) }
            var nextBlobId = 0
            
            for (y in 0 until height) {
                for (x in 0 until width) {
                    if (binaryImage[y][x] && !visited[y][x]) {
                        // Found a new blob, perform flood fill
                        val (blobSize, sumX, sumY) = floodFill(binaryImage, visited, x, y, width, height)
                        
                        // Only track blobs in reasonable size range to avoid noise and large objects
                        if (blobSize in MIN_BLOB_SIZE..MAX_BLOB_SIZE) {
                            newPlayerBlobs.add(
                                Blob(
                                    x = sumX / blobSize, // Average x position
                                    y = sumY / blobSize, // Average y position
                                    size = blobSize,
                                    id = nextBlobId++
                                )
                            )
                        }
                    }
                }
            }
            
            // Step 3: Track blobs across frames to detect movement
            if (previousPlayerBlobs.isNotEmpty() && newPlayerBlobs.isNotEmpty()) {
                // Calculate overall movement direction
                var leftMovement = 0
                var rightMovement = 0
                
                // Simple association based on proximity
                for (prevBlob in previousPlayerBlobs) {
                    // Find the closest new blob
                    val closestNewBlob = newPlayerBlobs.minByOrNull { 
                        squaredDistance(prevBlob.x, prevBlob.y, it.x, it.y) 
                    }
                    
                    // If found and reasonably close
                    closestNewBlob?.let { newBlob ->
                        val distance = squaredDistance(prevBlob.x, prevBlob.y, newBlob.x, newBlob.y)
                        if (distance < 2500) { // Within ~50 pixels
                            // Detect horizontal movement
                            val xDiff = newBlob.x - prevBlob.x
                            if (xDiff > 3) rightMovement++ // Moving right
                            else if (xDiff < -3) leftMovement++ // Moving left
                        }
                    }
                }
                
                // Determine predominant direction
                val newDirection = when {
                    rightMovement > leftMovement + 1 -> MovementDirection.RIGHT
                    leftMovement > rightMovement + 1 -> MovementDirection.LEFT
                    else -> movementDirection // Keep current direction if no clear winner
                }
                
                // Update direction with confidence tracking
                if (newDirection == movementDirection) {
                    directionConfidence = minOf(directionConfidence + 1, DIRECTION_CONFIDENCE_THRESHOLD + 5)
                } else {
                    directionConfidence--
                    if (directionConfidence <= 0) {
                        movementDirection = newDirection
                        directionConfidence = 1
                    }
                }
            }
            
            // Update for next frame
            previousPlayerBlobs = newPlayerBlobs
        }
        
        // Draw the processed image with players highlighted and direction indicator
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = result.getPixel(x, y)
                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)
                
                // Use the same improved green detection here
                val isGreen = isFieldGreen(r, g, b)
                val isBelowHorizon = y > height / 3
                val isAboveSideline = y < sidelineY
                
                if (isBelowHorizon && isAboveSideline) {
                    result.setPixel(x, y, if (isGreen) Color.WHITE else Color.BLACK)
                } else {
                    result.setPixel(x, y, if (isGreen) Color.LTGRAY else Color.DKGRAY)
                }
                
                if (y == height / 3) {
                    result.setPixel(x, y, Color.BLUE)
                }
                
                if (y == sidelineY) {
                    result.setPixel(x, y, Color.RED)
                }
            }
        }
        
        // Draw player blobs with color-coded highlights
        for (blob in newPlayerBlobs) {
            drawRect(result, blob.x - 5, blob.y - 5, blob.x + 5, blob.y + 5, Color.YELLOW)
        }
        
        // Add movement direction indicator
        val directionText = when (movementDirection) {
            MovementDirection.LEFT -> "← LEFT"
            MovementDirection.RIGHT -> "RIGHT →"
            MovementDirection.NONE -> "---"
        }
        
        if (directionConfidence >= DIRECTION_CONFIDENCE_THRESHOLD) {
            drawTextOnBitmap(result, directionText, width / 2, 20, 
                if (movementDirection == MovementDirection.LEFT) Color.CYAN else Color.MAGENTA)
        }
        
        // Fix rotation: Rotate 90 degrees to correct landscape orientation
        val rotatedBitmap = rotateBitmap(result, 90f)
        return rotatedBitmap
    }
    
    // Improved green detection with even more permissive approach
    private fun isFieldGreen(r: Int, g: Int, b: Int): Boolean {
        // Don't require all conditions to be met - any one of these approaches can classify as green
        
        // Primary approach with very expanded thresholds
        val standardGreen = (R_MIN <= r && r <= R_MAX &&
                G_MIN <= g && g <= G_MAX &&
                B_MIN <= b && b <= B_MAX &&
                g >= r + MIN_G_MINUS_R_DIFF &&
                g >= b + MIN_G_MINUS_B_DIFF)
        
        // Approach for more yellowish or dried grass - more permissive
        val yellowishGreen = (g > 70 && r > 70 && g >= r - 10 && g + r > b * 1.5)
        
        // Approach for darker artificial turf - more permissive
        val darkGreen = (g > 50 && g > r * 1.05 && g > b * 1.2 && r < 120 && b < 120)
        
        // Additional approach for light conditions where green is dominant
        val dominantGreen = (g > 100 && g > Math.max(r, b) * 0.9)
        
        // Approach for bluish artificial turf (some fields appear more blue-green)
        val bluishGreen = (g > 80 && b > 60 && g >= b - 20 && g + b > r * 1.8)
        
        return standardGreen || yellowishGreen || darkGreen || dominantGreen || bluishGreen
    }
    
    private fun floodFill(
        binary: Array<BooleanArray>, 
        visited: Array<BooleanArray>, 
        startX: Int, 
        startY: Int, 
        width: Int, 
        height: Int
    ): Triple<Int, Int, Int> {
        var size = 0
        var sumX = 0
        var sumY = 0
        
        val queue = java.util.ArrayDeque<Pair<Int, Int>>()
        queue.add(Pair(startX, startY))
        visited[startY][startX] = true
        
        while (queue.isNotEmpty()) {
            val (x, y) = queue.poll()
            size++
            sumX += x
            sumY += y
            
            val directions = arrayOf(
                Pair(1, 0), Pair(-1, 0), Pair(0, 1), Pair(0, -1)
            )
            
            for ((dx, dy) in directions) {
                val nx = x + dx
                val ny = y + dy
                
                if (nx in 0 until width && ny in 0 until height && 
                    binary[ny][nx] && !visited[ny][nx]) {
                    queue.add(Pair(nx, ny))
                    visited[ny][nx] = true
                }
            }
        }
        
        return Triple(size, sumX, sumY)
    }
    
    private fun squaredDistance(x1: Int, y1: Int, x2: Int, y2: Int): Int {
        val dx = x2 - x1
        val dy = y2 - y1
        return dx * dx + dy * dy
    }
    
    private fun drawRect(bitmap: Bitmap, left: Int, top: Int, right: Int, bottom: Int, color: Int) {
        val safeLeft = maxOf(0, minOf(left, bitmap.width - 1))
        val safeTop = maxOf(0, minOf(top, bitmap.height - 1))
        val safeRight = maxOf(0, minOf(right, bitmap.width - 1))
        val safeBottom = maxOf(0, minOf(bottom, bitmap.height - 1))
        
        for (x in safeLeft..safeRight) {
            if (safeTop in 0 until bitmap.height) bitmap.setPixel(x, safeTop, color)
            if (safeBottom in 0 until bitmap.height) bitmap.setPixel(x, safeBottom, color)
        }
        
        for (y in safeTop..safeBottom) {
            if (safeLeft in 0 until bitmap.width) bitmap.setPixel(safeLeft, y, color)
            if (safeRight in 0 until bitmap.width) bitmap.setPixel(safeRight, y, color)
        }
    }
    
    private fun drawTextOnBitmap(bitmap: Bitmap, text: String, x: Int, y: Int, color: Int) {
        val startX = maxOf(0, minOf(x - text.length * 3, bitmap.width - text.length * 6))
        val startY = maxOf(0, minOf(y, bitmap.height - 1))
        
        drawRect(bitmap, startX - 2, startY - 10, startX + text.length * 6 + 2, startY + 2, Color.BLACK)
        
        var posX = startX
        for (c in text) {
            when (c) {
                'L' -> drawL(bitmap, posX, startY, color)
                'E' -> drawE(bitmap, posX, startY, color)
                'F' -> drawF(bitmap, posX, startY, color)
                'T' -> drawT(bitmap, posX, startY, color)
                'R' -> drawR(bitmap, posX, startY, color)
                'I' -> drawI(bitmap, posX, startY, color)
                'G' -> drawG(bitmap, posX, startY, color)
                'H' -> drawH(bitmap, posX, startY, color)
                '←' -> drawLeftArrow(bitmap, posX, startY, color)
                '→' -> drawRightArrow(bitmap, posX, startY, color)
                '-' -> drawDash(bitmap, posX, startY, color)
                ' ' -> {} // Skip spaces
                else -> drawUnknown(bitmap, posX, startY, color)
            }
            posX += 6
        }
    }
    
    private fun drawL(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        for (dy in -8..0) {
            bitmap.safeSetPixel(x, y + dy, color)
        }
        for (dx in 0..4) {
            bitmap.safeSetPixel(x + dx, y, color)
        }
    }

    private fun drawE(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        for (dy in -8..0) {
            bitmap.safeSetPixel(x, y + dy, color)
        }
        for (dx in 0..4) {
            bitmap.safeSetPixel(x + dx, y, color)
            bitmap.safeSetPixel(x + dx, y - 4, color)
            bitmap.safeSetPixel(x + dx, y - 8, color)
        }
    }

    private fun drawF(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        for (dy in -8..0) {
            bitmap.safeSetPixel(x, y + dy, color)
        }
        for (dx in 0..4) {
            bitmap.safeSetPixel(x + dx, y - 4, color)
            bitmap.safeSetPixel(x + dx, y - 8, color)
        }
    }

    private fun drawT(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        for (dy in -8..0) {
            bitmap.safeSetPixel(x + 2, y + dy, color)
        }
        for (dx in 0..4) {
            bitmap.safeSetPixel(x + dx, y - 8, color)
        }
    }

    private fun drawR(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        for (dy in -8..0) {
            bitmap.safeSetPixel(x, y + dy, color)
        }
        bitmap.safeSetPixel(x + 1, y - 8, color)
        bitmap.safeSetPixel(x + 2, y - 8, color)
        bitmap.safeSetPixel(x + 3, y - 7, color)
        bitmap.safeSetPixel(x + 4, y - 6, color)
        bitmap.safeSetPixel(x + 3, y - 5, color)
        bitmap.safeSetPixel(x + 2, y - 4, color)
        bitmap.safeSetPixel(x + 3, y - 3, color)
        bitmap.safeSetPixel(x + 4, y - 1, color)
        bitmap.safeSetPixel(x + 4, y, color)
    }

    private fun drawI(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        for (dy in -8..0) {
            bitmap.safeSetPixel(x + 2, y + dy, color)
        }
    }

    private fun drawG(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        for (dy in -7..0) {
            bitmap.safeSetPixel(x, y + dy - 1, color)
        }
        for (dx in 0..4) {
            bitmap.safeSetPixel(x + dx, y, color)
            bitmap.safeSetPixel(x + dx, y - 8, color)
        }
        bitmap.safeSetPixel(x + 4, y - 1, color)
        bitmap.safeSetPixel(x + 4, y - 2, color)
        bitmap.safeSetPixel(x + 3, y - 3, color)
        bitmap.safeSetPixel(x + 2, y - 3, color)
    }

    private fun drawH(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        for (dy in -8..0) {
            bitmap.safeSetPixel(x, y + dy, color)
            bitmap.safeSetPixel(x + 4, y + dy, color)
        }
        for (dx in 0..4) {
            bitmap.safeSetPixel(x + dx, y - 4, color)
        }
    }

    private fun drawLeftArrow(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        bitmap.safeSetPixel(x + 4, y - 4, color)
        bitmap.safeSetPixel(x + 3, y - 3, color)
        bitmap.safeSetPixel(x + 2, y - 2, color)
        bitmap.safeSetPixel(x + 1, y - 1, color)
        bitmap.safeSetPixel(x, y, color)
        bitmap.safeSetPixel(x + 1, y + 1, color)
        bitmap.safeSetPixel(x + 2, y + 2, color)
        bitmap.safeSetPixel(x + 3, y + 3, color)
        bitmap.safeSetPixel(x + 4, y + 4, color)
    }

    private fun drawRightArrow(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        bitmap.safeSetPixel(x, y - 4, color)
        bitmap.safeSetPixel(x + 1, y - 3, color)
        bitmap.safeSetPixel(x + 2, y - 2, color)
        bitmap.safeSetPixel(x + 3, y - 1, color)
        bitmap.safeSetPixel(x + 4, y, color)
        bitmap.safeSetPixel(x + 3, y + 1, color)
        bitmap.safeSetPixel(x + 2, y + 2, color)
        bitmap.safeSetPixel(x + 1, y + 3, color)
        bitmap.safeSetPixel(x, y + 4, color)
    }

    private fun drawDash(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        for (dx in 0..4) {
            bitmap.safeSetPixel(x + dx, y - 4, color)
        }
    }

    private fun drawUnknown(bitmap: Bitmap, x: Int, y: Int, color: Int) {
        // Draw a simple rectangle for characters we don't have specific drawing functions for
        drawRect(bitmap, x, y - 8, x + 4, y, color)
    }

    private fun Bitmap.safeSetPixel(x: Int, y: Int, color: Int) {
        if (x in 0 until width && y in 0 until height) {
            setPixel(x, y, color)
        }
    }
    
    private fun toBitmap(imageProxy: ImageProxy): Bitmap? {
        val image = imageProxy.image ?: return null
        
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        
        val nv21 = ByteArray(ySize + uSize + vSize)
        
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        
        val yuvImage = android.graphics.YuvImage(
            nv21, android.graphics.ImageFormat.NV21, imageProxy.width, imageProxy.height, null
        )
        
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, imageProxy.width, imageProxy.height), 80, out)
        val imageBytes = out.toByteArray()
        
        val options = android.graphics.BitmapFactory.Options().apply {
            inSampleSize = 2
        }
        
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, options)
    }
    
    private fun rotateBitmap(source: Bitmap, degrees: Float): Bitmap {
        val matrix = android.graphics.Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(
            source, 0, 0, source.width, source.height, matrix, true
        )
    }
    
    fun release() {
        backgroundExecutor.shutdown()
        labeler.close()
    }
}
