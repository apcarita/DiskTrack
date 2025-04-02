package com.example.disktrack

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.disktrack.ui.theme.DiskTrackTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private const val TAG = "CameraDebugActivity"

class CameraDebugActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DiskTrackTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    CameraDebugScreen()
                }
            }
        }
    }
}

@Composable
fun CameraDebugScreen() {
    val context = LocalContext.current
    var cameraInfo by remember { mutableStateOf("Loading camera information...") }
    var permissionStatus by remember { mutableStateOf("Checking permissions...") }
    var cameraXLibInfo by remember { mutableStateOf("Checking CameraX...") }
    
    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            try {
                // Check camera permission
                val hasCameraPermission = ContextCompat.checkSelfPermission(
                    context, 
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
                
                permissionStatus = "Camera Permission: ${if (hasCameraPermission) "GRANTED" else "DENIED"}"
                
                // Get camera info
                val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                val cameraList = cameraManager.cameraIdList
                
                val cameraDetails = StringBuilder()
                cameraDetails.appendLine("Found ${cameraList.size} cameras:")
                
                cameraList.forEach { id ->
                    val characteristics = cameraManager.getCameraCharacteristics(id)
                    val facing = characteristics.get(android.hardware.camera2.CameraCharacteristics.LENS_FACING)
                    val facingStr = when (facing) {
                        android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT -> "Front"
                        android.hardware.camera2.CameraCharacteristics.LENS_FACING_BACK -> "Back"
                        else -> "External"
                    }
                    cameraDetails.appendLine("Camera $id: $facingStr")
                }
                
                cameraInfo = cameraDetails.toString()
                
                // Check CameraX libraries
                try {
                    val previewClass = Class.forName("androidx.camera.core.Preview")
                    val cameraProviderClass = Class.forName("androidx.camera.lifecycle.ProcessCameraProvider")
                    cameraXLibInfo = "CameraX libraries appear to be properly loaded"
                } catch (e: Exception) {
                    cameraXLibInfo = "Error loading CameraX classes: ${e.message}"
                }
                
            } catch (e: Exception) {
                cameraInfo = "Error getting camera info: ${e.message}"
                Log.e(TAG, "Failed to get camera info", e)
            }
        }
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            "Camera Debug Information",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        
        DebugSection("Permission Status") {
            Text(permissionStatus)
        }
        
        DebugSection("Camera Information") {
            Text(cameraInfo)
        }
        
        DebugSection("CameraX Library Status") {
            Text(cameraXLibInfo)
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Button(onClick = {
            // You could add functionality to test the camera directly
        }) {
            Text("Test Camera Preview")
        }
    }
}

@Composable
fun DebugSection(title: String, content: @Composable () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    ) {
        Text(
            title,
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier
                .background(Color.LightGray)
                .padding(4.dp)
                .fillMaxWidth()
        )
        
        Spacer(modifier = Modifier.height(4.dp))
        
        content()
        
        Divider(modifier = Modifier.padding(top = 8.dp))
    }
}
