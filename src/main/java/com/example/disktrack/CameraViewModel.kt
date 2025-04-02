package com.example.disktrack

import android.content.Context
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraViewModel : ViewModel() {
    private val _cameraState = MutableStateFlow<CameraState>(CameraState.Initializing)
    val cameraState = _cameraState.asStateFlow()

    private lateinit var cameraExecutor: ExecutorService

    init {
        // Initialize camera executor in init block to ensure it's created
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    // Remove the initialize function as we're now initializing in init block
    
    fun startCamera(
        context: Context,
        lifecycleOwner: LifecycleOwner,
        previewView: Preview.SurfaceProvider
    ) {
        _cameraState.value = CameraState.Initializing
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                // Get camera provider
                val cameraProvider = cameraProviderFuture.get()
                
                // Unbind any existing use cases
                cameraProvider.unbindAll()
                
                // Create preview use case
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView)
                }
                
                // Use the rear camera by default
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                
                // Bind use cases to lifecycle
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview
                )
                
                _cameraState.value = CameraState.Preview
            } catch (e: Exception) {
                _cameraState.value = CameraState.Error(e.message ?: "Unknown error")
            }
        }, ContextCompat.getMainExecutor(context))
    }
    
    override fun onCleared() {
        super.onCleared()
        if (::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
        }
    }
    
    sealed class CameraState {
        object Initializing : CameraState()
        object Preview : CameraState()
        data class Error(val message: String) : CameraState()
    }
}
