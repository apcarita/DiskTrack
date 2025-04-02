package com.example.disktrack

import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.zIndex
import androidx.core.content.ContextCompat
import java.util.concurrent.Executor
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

private const val TAG = "CameraPreviewContent"

@Composable
fun CameraPreviewContent(modifier: Modifier = Modifier) {
    // Set up all the necessary objects for camera
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val executor = remember { ContextCompat.getMainExecutor(context) }
    
    // Debug state tracking
    var debugState by remember { mutableStateOf("Initializing...") }
    var hasError by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf("") }
    
    // Remember the PreviewView
    val previewView = remember {
        PreviewView(context).apply {
            this.scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    
    // Set up the camera separately from AndroidView
    LaunchedEffect(previewView) {
        debugState = "LaunchedEffect started"
        Log.d(TAG, "LaunchedEffect started for camera setup")
        
        try {
            debugState = "Getting camera provider"
            Log.d(TAG, "Getting camera provider")
            
            val cameraProvider = getCameraProvider(context, executor)
            debugState = "Camera provider obtained"
            Log.d(TAG, "Camera provider successfully obtained")
            
            // Unbind any previous use cases
            cameraProvider.unbindAll()
            debugState = "Unbound previous use cases"
            Log.d(TAG, "Unbound previous camera use cases")
            
            // Create the Preview use case
            debugState = "Building preview"
            val preview = Preview.Builder()
                .build()
                .also {
                    debugState = "Setting surface provider"
                    Log.d(TAG, "Setting surface provider to PreviewView")
                    // Attach the preview to the PreviewView's surface
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
            
            // Select the back camera
            debugState = "Selecting back camera"
            Log.d(TAG, "Selecting back camera")
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            // Bind the camera to the lifecycle
            debugState = "Binding to lifecycle"
            Log.d(TAG, "Binding camera to lifecycle")
            
            val camera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview
            )
            
            debugState = "Camera setup complete"
            Log.d(TAG, "Camera successfully bound to lifecycle: ${camera.cameraInfo}")
        } catch (e: Exception) {
            hasError = true
            errorMessage = e.message ?: "Unknown error"
            debugState = "Camera setup failed"
            Log.e(TAG, "Camera initialization failed", e)
            e.printStackTrace()
        }
    }
    
    // This box will contain our camera preview and any overlays
    Box(modifier = modifier.fillMaxSize()) {
        // Debug overlay panel 
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.Black.copy(alpha = 0.7f))
                .padding(8.dp)
                .align(Alignment.TopCenter)
                .zIndex(2f),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Camera Debug: $debugState",
                color = if (hasError) Color.Red else Color.Green,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(4.dp)
            )
            
            if (hasError) {
                Text(
                    text = errorMessage,
                    color = Color.Red,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(4.dp)
                )
            }
        }
        
        // Create the AndroidView for the PreviewView
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { previewView }
        )
        
        // Show a loading indicator while the camera is initializing
        if (debugState.contains("Initializing") || debugState.contains("Creating") || debugState.contains("Getting")) {
            CircularProgressIndicator(
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(bottom = 100.dp)
            )
        }
    }
}

// Helper function to get the camera provider
private suspend fun getCameraProvider(context: android.content.Context, executor: Executor): ProcessCameraProvider {
    Log.d(TAG, "Getting ProcessCameraProvider instance")
    return suspendCoroutine { continuation ->
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        Log.d(TAG, "Adding listener to camera provider future")
        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                Log.d(TAG, "ProcessCameraProvider obtained successfully: $provider")
                continuation.resume(provider)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get camera provider", e)
                throw e
            }
        }, executor)
    }
}
