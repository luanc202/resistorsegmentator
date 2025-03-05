package br.ufma.resistorsegmentation

import android.content.pm.PackageManager
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.graphics.get
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlayView: SegmentationOverlayView
    private lateinit var textViewResult: TextView
    private lateinit var resistorClassifier: ResistorClassifier
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.preview_view)
        overlayView = findViewById(R.id.overlay_view)
        textViewResult = findViewById(R.id.text_view_result)
        resistorClassifier = ResistorClassifier(this)

        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 100)
        } else {
            startCamera()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        val bitmap = imageProxy.toBitmap()
                        Log.i("MainActivity", "Bitmap: ")
                        Log.i("MainActivity", bitmap.height.toString())
                        Log.i("MainActivity", bitmap.width.toString())
                        val results = resistorClassifier.classify(bitmap)

//                        Log.i("MainActivity", results.toString())

                        val boundingBoxes = results.map {
                            // Simula uma bounding box simples (ajuste conforme necessário)
                            RectF(100f, 100f, 300f, 200f)
                        }

                        val segmentationResults = results.mapIndexed { index, (label, score) ->
                            SegmentationResult(
                                boundingBoxes[index],
                                "$label: ${"%.2f".format(score * 100)}%",
                                bitmap // Usando bitmap como máscara temporária
                            )
                        }

                        runOnUiThread {
                            textViewResult.text = "Detectado:\n${results.joinToString("\n") {
                                    (label, score) -> "$label: ${"%.2f".format(score * 100)}%"
                            }}"
                            overlayView.setSegmentationResults(segmentationResults)
                        }
                        imageProxy.close()
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}