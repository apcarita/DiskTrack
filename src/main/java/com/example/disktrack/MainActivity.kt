package com.example.disktrack

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleEventObserver
import com.example.disktrack.ui.theme.DiskTrackTheme
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionState
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale

private const val TAG = "MainActivity"

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "MainActivity onCreate")
        
        // Check if camera permission is already granted
        val hasCameraPermission = ContextCompat.checkSelfPermission(
            this, 
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        
        Log.d(TAG, "Initial camera permission status: $hasCameraPermission")
        
        enableEdgeToEdge()
        setContent {
            Log.d(TAG, "Setting content")
            DiskTrackTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    CameraApp(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
    
    override fun onResume() {
        super.onResume()
        Log.d(TAG, "MainActivity onResume")
    }
    
    override fun onPause() {
        super.onPause()
        Log.d(TAG, "MainActivity onPause")
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun CameraApp(modifier: Modifier = Modifier) {
    val lifecycleOwner = LocalLifecycleOwner.current
    
    // Monitor lifecycle events
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            Log.d(TAG, "Lifecycle event: $event")
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }
    
    val cameraPermissionState = rememberPermissionState(
        Manifest.permission.CAMERA,
        onPermissionResult = { isGranted ->
            Log.d(TAG, "Camera permission result: $isGranted")
        }
    )
    
    Log.d(TAG, "Current permission state: ${cameraPermissionState.status}")
    
    if (cameraPermissionState.status.isGranted) {
        Log.d(TAG, "Camera permission is granted, showing camera preview")
        CameraPreviewContent(modifier)
    } else {
        Log.d(TAG, "Camera permission not granted, showing permission request screen")
        RequestCameraPermission(cameraPermissionState, modifier)
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
private fun RequestCameraPermission(
    cameraPermissionState: PermissionState,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .wrapContentSize()
            .widthIn(max = 480.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        val textToShow = if (cameraPermissionState.status.shouldShowRationale) {
            "We need camera access to show the camera preview. " +
                "Please grant permission to continue."
        } else {
            "Camera access is required for this app to work. " +
                "Please grant the permission."
        }
        
        Text(
            textToShow, 
            textAlign = TextAlign.Center, 
            modifier = Modifier.padding(16.dp)
        )
        
        Spacer(Modifier.height(16.dp))
        
        Button(onClick = { 
            Log.d(TAG, "Requesting camera permission")
            cameraPermissionState.launchPermissionRequest() 
        }) {
            Text("Grant Camera Permission")
        }
    }
}

@Preview(showBackground = true)
@Composable
fun CameraAppPreview() {
    DiskTrackTheme {
        CameraApp()
    }
}