package com.example.disktrack

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.zIndex
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel

private const val TAG = "CameraPreviewContent"

@Composable
fun CameraPreviewContent(modifier: Modifier = Modifier, viewModel: CameraViewModel = viewModel()) {
    // Set up all the necessary objects for camera
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    
    // Track field visualization bitmap
    var fieldBitmap by remember { mutableStateOf<Bitmap?>(null) }
    
    // Debug state tracking
    var debugState by remember { mutableStateOf("Initializing...") }
    var hasError by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf("") }
    
    // Create analyzer that only produces the field visualization
    val imageAnalyzer = remember { 
        PlayerImageAnalyzer { bitmap ->
            Log.d(TAG, "Received new bitmap from analyzer: ${bitmap.width}x${bitmap.height}")
            fieldBitmap = bitmap
        }
    }
    
    // Observe camera state
    val cameraState by viewModel.cameraState.collectAsState()
    
    // Update debug state based on camera state
    LaunchedEffect(cameraState) {
        debugState = when (cameraState) {
            is CameraViewModel.CameraState.Initializing -> "Initializing camera..."
            is CameraViewModel.CameraState.Preview -> "Camera ready"
            is CameraViewModel.CameraState.Error -> {
                hasError = true
                errorMessage = (cameraState as CameraViewModel.CameraState.Error).message
                "Camera error"
            }
        }
        Log.d(TAG, "Camera state updated: $debugState")
    }
    
    // Remember the PreviewView
    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    
    // Initialize and configure camera
    LaunchedEffect(previewView) {
        try {
            Log.d(TAG, "Setting up camera...")
            
            // Create the camera provider
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            val cameraProvider = cameraProviderFuture.get()
            
            // Unbind any previous use cases
            cameraProvider.unbindAll()
            
            // Create preview use case
            val preview = Preview.Builder().build()
            preview.setSurfaceProvider(previewView.surfaceProvider)
            
            // Setup image analysis
            val imageAnalysisBuilder = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                
            val imageAnalysisUseCase = imageAnalysisBuilder.build()
            imageAnalysisUseCase.setAnalyzer(ContextCompat.getMainExecutor(context), imageAnalyzer)
            
            // Use the back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            // Bind camera to lifecycle
            val camera = cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalysisUseCase
            )
            
            Log.d(TAG, "Camera setup complete with exposure mode: ${camera.cameraInfo.exposureState}")
            debugState = "Camera ready"
            
        } catch (e: Exception) {
            hasError = true
            errorMessage = e.message ?: "Unknown error"
            debugState = "Camera setup failed: ${e.message}"
            Log.e(TAG, "Failed to start camera", e)
        }
    }

    // This box will contain our visualization
    Box(modifier = modifier.fillMaxSize()) {
        // We'll keep the Android view for the preview but make it invisible
        // This ensures the camera keeps working even though we don't show the preview
        AndroidView(
            factory = { previewView },
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Transparent)
        )
        
        // Show field visualization as main content
        fieldBitmap?.let { bitmap ->
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Field Visualization",
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black)
            )
        } ?: run {
            // Show a placeholder when no bitmap is available
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color.Black)
            )
        }
        
        // Status overlay panel
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
                text = "Field Analyzer",
                color = Color.White,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(4.dp)
            )
            
            Text(
                text = "Status: $debugState",
                color = if (hasError) Color.Red else Color.Green,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(4.dp)
            )

            Text(
                text = "Bitmap: ${fieldBitmap?.width ?: 0}x${fieldBitmap?.height ?: 0}",
                color = Color.White,
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
        
        // Only show loading indicator when the field bitmap is null
        if (fieldBitmap == null) {
            CircularProgressIndicator(
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(bottom = 100.dp)
            )
        }
    }
}
