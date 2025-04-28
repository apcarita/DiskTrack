package com.example.disktrack3

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.Rect // Use Rect for initialization and update output
import org.opencv.core.Rect2d
import org.opencv.tracking.TrackerMOSSE // Use MOSSE tracker from legacy package
import kotlin.math.max // Import necessary math functions explicitly if needed
import kotlin.math.min // Import necessary math functions explicitly if needed

class YoloTracker(
    private val maxMisses: Int = 10, // Max frames a track can be missed before deletion
    private val minIouThreshold: Double = 0.2 // Min IoU to associate detection with track
) {
    private val TAG = "YoloTracker"

    // Data class to hold information about each tracked object
    private data class Track(
        val id: Int,
        var tracker: TrackerMOSSE, // Use the specific tracker type here (legacy)
        var boundingBox: Rect2d, // Use Rect2d for tracker precision and updates
        var misses: Int = 0,
        var age: Int = 0,
        var className: String = "person" // Assuming we only track persons for now
    )

    private var activeTracks = mutableMapOf<Int, Track>()
    private var nextTrackId = 0

    // Calculates Intersection over Union (IoU) between two rectangles
    private fun calculateIoU(box1: Rect2d, box2: Rect2d): Double {
        val xA = maxOf(box1.x, box2.x)
        val yA = maxOf(box1.y, box2.y)
        val xB = minOf(box1.x + box1.width, box2.x + box2.width)
        val yB = minOf(box1.y + box1.height, box2.y + box2.height)

        val intersectionArea = maxOf(0.0, xB - xA) * maxOf(0.0, yB - yA)
        if (intersectionArea == 0.0) return 0.0

        val box1Area = box1.width * box1.height
        val box2Area = box2.width * box2.height

        val unionArea = box1Area + box2Area - intersectionArea
        return if (unionArea > 0) intersectionArea / unionArea else 0.0
    }

    // Converts YoloDetector.Detection Rect to Rect2d
    private fun detectionToRect2d(detection: YoloDetector.Detection): Rect2d {
        return Rect2d(
            detection.boundingBox.x.toDouble(),
            detection.boundingBox.y.toDouble(),
            detection.boundingBox.width.toDouble(),
            detection.boundingBox.height.toDouble()
        )
    }

    // Helper to convert Rect2d to Rect for initialization
    private fun rect2dToRect(rect2d: Rect2d): Rect {
        return Rect(rect2d.x.toInt(), rect2d.y.toInt(), rect2d.width.toInt(), rect2d.height.toInt())
    }

    // Helper to convert Rect to Rect2d
    private fun rectToRect2d(rect: Rect): Rect2d {
        return Rect2d(rect.x.toDouble(), rect.y.toDouble(), rect.width.toDouble(), rect.height.toDouble())
    }

    // Main update function called for each frame
    fun update(frame: Mat, detections: List<YoloDetector.Detection>?): List<Pair<Int, Rect>> {
        Log.d(TAG, "YoloTracker.update called | frame.empty=${frame.empty()}, detections=${detections?.size ?: 0}, activeTracks=${activeTracks.size}")
        if (frame.empty()) {
            Log.e(TAG, "Update called with empty frame!")
            activeTracks.values.forEach { it.misses++ }
            val trackIdsToRemoveOnEmpty = activeTracks.filter { it.value.misses > maxMisses }.keys.toList()
            trackIdsToRemoveOnEmpty.forEach { activeTracks.remove(it) }
            return activeTracks.values.map { track ->
                Pair(track.id, Rect(track.boundingBox.x.toInt(), track.boundingBox.y.toInt(), track.boundingBox.width.toInt(), track.boundingBox.height.toInt()))
            }
        }

        val predictedBoxes = mutableMapOf<Int, Rect2d>()
        val trackIdsToRemove = mutableListOf<Int>()

        // 1. Predict new locations for existing tracks AND increment misses
        for ((id, track) in activeTracks) {
            track.misses++

            var predictedBoxRect = rect2dToRect(track.boundingBox)

            predictedBoxRect.x = maxOf(0, predictedBoxRect.x)
            predictedBoxRect.y = maxOf(0, predictedBoxRect.y)
            predictedBoxRect.width = minOf(frame.cols() - predictedBoxRect.x, predictedBoxRect.width)
            predictedBoxRect.height = minOf(frame.rows() - predictedBoxRect.y, predictedBoxRect.height)

            predictedBoxRect.width = maxOf(1, predictedBoxRect.width)
            predictedBoxRect.height = maxOf(1, predictedBoxRect.height)

            if (predictedBoxRect.x + predictedBoxRect.width > frame.cols() || predictedBoxRect.y + predictedBoxRect.height > frame.rows()) {
                Log.e(TAG, "Track $id PREDICTED box $predictedBoxRect exceeds frame dims (${frame.cols()}x${frame.rows()}) even after clamping! Marking for removal.")
                trackIdsToRemove.add(id)
                continue
            }

            Log.d(TAG, "Track $id: Updating with frame (${frame.cols()}x${frame.rows()}) and clamped box $predictedBoxRect (Misses: ${track.misses})")

            val success: Boolean
            try {
                success = track.tracker.update(frame, predictedBoxRect)
            } catch (e: Exception) {
                Log.e(TAG, "Exception during tracker update for track $id: ${e.message}", e)
                trackIdsToRemove.add(id)
                continue
            }

            if (!success) {
                Log.d(TAG, "Track $id update FAILED. Marking for removal.")
                trackIdsToRemove.add(id)
                continue
            }

            if (predictedBoxRect.width > 0 && predictedBoxRect.height > 0) {
                predictedBoxRect.x = maxOf(0, predictedBoxRect.x)
                predictedBoxRect.y = maxOf(0, predictedBoxRect.y)
                predictedBoxRect.width = minOf(frame.cols() - predictedBoxRect.x, predictedBoxRect.width)
                predictedBoxRect.height = minOf(frame.rows() - predictedBoxRect.y, predictedBoxRect.height)

                predictedBoxRect.width = maxOf(1, predictedBoxRect.width)
                predictedBoxRect.height = maxOf(1, predictedBoxRect.height)

                if (predictedBoxRect.width > 0 && predictedBoxRect.height > 0) {
                    val predictedBox2d = rectToRect2d(predictedBoxRect)
                    track.boundingBox = predictedBox2d
                    predictedBoxes[id] = predictedBox2d
                    track.age++
                    Log.d(TAG, "Track $id update SUCCESS. New box: $predictedBoxRect")
                } else {
                    Log.w(TAG, "Track $id prediction resulted in zero/negative size box after update+clamping. Marking for removal.")
                    trackIdsToRemove.add(id)
                }
            } else {
                Log.w(TAG, "Track $id prediction resulted in zero/negative size box immediately after update. Marking for removal.")
                trackIdsToRemove.add(id)
            }
        }

        if (trackIdsToRemove.isNotEmpty()) {
            Log.d(TAG, "Removing ${trackIdsToRemove.size} tracks due to update failure or invalid state.")
            trackIdsToRemove.forEach { id ->
                activeTracks.remove(id)
                Log.d(TAG, "Removed track $id.")
            }
        }

        Log.d(TAG, "After prediction/removal | predictedBoxes=${predictedBoxes.size}, activeTracks=${activeTracks.size}")

        var matchedDetectionIndices = mutableSetOf<Int>()
        val unmatchedDetectionIndices = mutableListOf<Int>()

        if (detections != null && detections.isNotEmpty()) {
            Log.d(TAG, "Associating ${detections.size} detections with ${predictedBoxes.size} predicted boxes")

            unmatchedDetectionIndices.addAll(detections.indices)

            if (predictedBoxes.isNotEmpty()) {
                val detectionBoxes = detections.map { detectionToRect2d(it) }
                val numTracks = predictedBoxes.size
                val numDetections = detectionBoxes.size
                val iouMatrix = Array(numTracks) { DoubleArray(numDetections) { 0.0 } }
                val trackIdList = predictedBoxes.keys.toList()

                for (i in 0 until numTracks) {
                    val trackId = trackIdList[i]
                    val predictedBox = predictedBoxes[trackId]
                    if (predictedBox != null) {
                        for (j in 0 until numDetections) {
                            iouMatrix[i][j] = calculateIoU(predictedBox, detectionBoxes[j])
                        }
                    } else {
                        Log.w(TAG, "Predicted box for track $trackId was null during IoU calculation.")
                    }
                }

                val trackMatches = mutableMapOf<Int, Int>()
                val detectionMatched = BooleanArray(numDetections) { false }

                for (i in 0 until numTracks) {
                    val trackId = trackIdList[i]
                    var bestIoU = minIouThreshold
                    var bestDetectionIndex = -1

                    for (j in 0 until numDetections) {
                        if (!detectionMatched[j] && iouMatrix[i][j] > bestIoU) {
                            bestIoU = iouMatrix[i][j]
                            bestDetectionIndex = j
                        }
                    }

                    if (bestDetectionIndex != -1) {
                        trackMatches[trackId] = bestDetectionIndex
                        detectionMatched[bestDetectionIndex] = true
                        matchedDetectionIndices.add(bestDetectionIndex)
                        unmatchedDetectionIndices.remove(bestDetectionIndex)
                        Log.d(TAG, "Matched Track ${trackId} (row $i) with Detection $bestDetectionIndex (IoU: ${"%.2f".format(bestIoU)})")

                        val matchedTrack = activeTracks[trackId]
                        val matchedDetection = detections[bestDetectionIndex]
                        if (matchedTrack != null) {
                            val detectionBox2d = detectionToRect2d(matchedDetection)
                            matchedTrack.boundingBox = detectionBox2d
                            matchedTrack.misses = 0
                            matchedTrack.className = matchedDetection.className
                            Log.d(TAG, "Updated track $trackId position based on detection $bestDetectionIndex")
                        }
                    } else {
                        Log.d(TAG, "Track $trackId found no suitable detection match.")
                    }
                }
                Log.d(TAG, "Matched detection indices: $matchedDetectionIndices")
            } else {
                Log.d(TAG, "No active tracks to match detections against.")
            }

            Log.d(TAG, "Unmatched detection indices after matching: $unmatchedDetectionIndices")

            if (unmatchedDetectionIndices.isNotEmpty()) {
                Log.d(TAG, "Found ${unmatchedDetectionIndices.size} unmatched detections to create tracks for.")
            }

            for (j in unmatchedDetectionIndices) {
                if (j < 0 || j >= detections.size) {
                    Log.e(TAG, "Invalid unmatched index $j encountered.")
                    continue
                }
                val detection = detections[j]
                if (detection.boundingBox.width <= 0 || detection.boundingBox.height <= 0) {
                    Log.w(TAG, "Skipping new track for detection $j due to invalid bounding box: ${detection.boundingBox}")
                    continue
                }

                val newTracker = TrackerMOSSE.create()
                val detectionBox2d = detectionToRect2d(detection)
                var detectionBoxRect = rect2dToRect(detectionBox2d)

                detectionBoxRect.x = maxOf(0, detectionBoxRect.x)
                detectionBoxRect.y = maxOf(0, detectionBoxRect.y)
                detectionBoxRect.width = minOf(frame.cols() - detectionBoxRect.x, detectionBoxRect.width)
                detectionBoxRect.height = minOf(frame.rows() - detectionBoxRect.y, detectionBoxRect.height)
                detectionBoxRect.width = maxOf(1, detectionBoxRect.width)
                detectionBoxRect.height = maxOf(1, detectionBoxRect.height)

                Log.d(TAG, "Attempting init for new track (detection $j) at $detectionBoxRect")
                if (detectionBoxRect.width <= 0 || detectionBoxRect.height <= 0) {
                    Log.e(TAG, "Cannot init tracker for detection $j, box became invalid after clamping: $detectionBoxRect")
                    continue
                }

                val initSuccess = try {
                    newTracker.init(frame, detectionBoxRect)
                    true
                } catch (e: Exception) {
                    Log.e(TAG, "Exception during new tracker initialization for detection $j: ${e.message}", e)
                    false
                }
                Log.d(TAG, "Init success for new track (detection $j): $initSuccess")

                if (initSuccess) {
                    val newId = nextTrackId++
                    val initialBox2d = rectToRect2d(detectionBoxRect)
                    val newTrack = Track(
                        id = newId,
                        tracker = newTracker,
                        boundingBox = initialBox2d,
                        className = detection.className
                    )
                    activeTracks[newId] = newTrack
                    Log.d(TAG, "Created new track $newId for unmatched detection $j (${detection.className})")
                } else {
                    Log.e(TAG, "Failed to initialize tracker for new detection $j")
                }
            }
        } else {
            Log.d(TAG, "No detections provided for association.")
        }

        val finalTrackIdsToRemove = mutableListOf<Int>()
        for ((id, track) in activeTracks) {
            if (track.misses > maxMisses) {
                finalTrackIdsToRemove.add(id)
                Log.d(TAG, "Marking track $id for final removal due to max misses (${track.misses} > $maxMisses).")
            }
        }
        if (finalTrackIdsToRemove.isNotEmpty()) {
            Log.d(TAG, "Removing ${finalTrackIdsToRemove.size} tracks due to max misses.")
            finalTrackIdsToRemove.forEach { id ->
                activeTracks.remove(id)
                Log.d(TAG, "Removed track $id.")
            }
        }

        Log.d(TAG, "Returning ${activeTracks.size} active tracks")
        return activeTracks.values
            .filter { it.boundingBox.width > 0 && it.boundingBox.height > 0 }
            .map { track ->
                Pair(track.id, Rect(track.boundingBox.x.toInt(), track.boundingBox.y.toInt(), track.boundingBox.width.toInt(), track.boundingBox.height.toInt()))
            }
    }

    // Optional: Method to clear all tracks
    fun clear() {
        activeTracks.clear()
        nextTrackId = 0
        Log.d(TAG, "Cleared all tracks map.")
    }

    // Initialize trackers for new detections on the given grayscale frame
    fun initWithDetections(initFrame: Mat, detections: List<YoloDetector.Detection>) {
        Log.d(TAG, "initWithDetections called with ${detections.size} detections")
        if (initFrame.empty()) {
            Log.e(TAG, "initWithDetections called with empty frame! Cannot initialize.")
            return
        }
        clear()
        detections.forEachIndexed { idx, det ->
            if (det.boundingBox.width <= 0 || det.boundingBox.height <= 0) {
                Log.w(TAG, "Skipping init for detection $idx due to invalid bounding box: ${det.boundingBox}")
                return@forEachIndexed
            }

            val newTracker = TrackerMOSSE.create()
            val box2d = detectionToRect2d(det)
            var boxRect = rect2dToRect(box2d)

            boxRect.x = maxOf(0, boxRect.x)
            boxRect.y = maxOf(0, boxRect.y)
            boxRect.width = minOf(initFrame.cols() - boxRect.x, boxRect.width)
            boxRect.height = minOf(initFrame.rows() - boxRect.y, boxRect.height)
            boxRect.width = maxOf(1, boxRect.width)
            boxRect.height = maxOf(1, boxRect.height)

            Log.d(TAG, "initWithDetections: init track for det $idx at $boxRect")
            if (boxRect.width <= 0 || boxRect.height <= 0) {
                Log.e(TAG, "Cannot init tracker for detection $idx, box became invalid after clamping: $boxRect")
                return@forEachIndexed
            }

            val ok = try {
                newTracker.init(initFrame, boxRect)
                true
            } catch (e: Exception) {
                Log.e(TAG, "initWithDetections: failed det $idx: ${e.message}", e)
                false
            }
            if (ok) {
                val id = nextTrackId++
                val initialBox2d = rectToRect2d(boxRect)
                activeTracks[id] = Track(id, newTracker, initialBox2d, className = det.className, misses = 0)
                Log.d(TAG, "initWithDetections: added track $id")
            } else {
                Log.e(TAG, "initWithDetections: failed to init tracker for det $idx")
            }
        }
    }
}
