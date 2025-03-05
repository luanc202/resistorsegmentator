package br.ufma.resistorsegmentation

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import br.ufma.resistorsegmentation.types.SegmentationResult
import com.davemorrissey.labs.subscaleview.ImageSource
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import com.davemorrissey.labs.subscaleview.SubsamplingScaleImageView

class MainActivity : ComponentActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var resultImageView: SubsamplingScaleImageView
    private lateinit var overlayView: SegmentationOverlayView
    private lateinit var captureButton: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var imageCapture: ImageCapture
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var cameraProvider: ProcessCameraProvider
    private val analyzer by lazy { Analyzer(this) }
    private var isCameraActive = false
    private val scope = CoroutineScope(Dispatchers.Main)
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        style = Paint.Style.FILL
        setShadowLayer(2f, 2f, 2f, Color.BLACK) // Add shadow for readability
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.preview_view)
        resultImageView = findViewById(R.id.result_image_view)
        overlayView = findViewById(R.id.overlay_view)
        captureButton = findViewById(R.id.capture_button)
        progressBar = findViewById(R.id.progress_bar)
        cameraExecutor = Executors.newSingleThreadExecutor()

        if (checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission()
        } else {
            startCamera()
        }

        captureButton.setOnClickListener {
            if (isCameraActive) {
                takePhoto()
            } else {
                resetToCamera()
            }
        }
    }

    private fun requestCameraPermission() {
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startCamera()
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show()
            }
        }.launch(android.Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageCapture
            )
            isCameraActive = true
            previewView.visibility = View.VISIBLE
            resultImageView.visibility = View.GONE
            overlayView.visibility = View.VISIBLE
            captureButton.text = "Capture Photo"
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val outputFile = File(externalMediaDirs.first(), "${System.currentTimeMillis()}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(outputFile).build()

        imageCapture.takePicture(
            outputOptions,
            cameraExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val bitmap = android.graphics.BitmapFactory.decodeFile(outputFile.absolutePath)
                    scope.launch {
                        cameraProvider.unbindAll()
                        isCameraActive = false
                        runInference(bitmap)
                    }
                    outputFile.delete()
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e("MainActivity", "Photo capture failed: ${exception.message}", exception)
                    scope.launch {
                        Toast.makeText(this@MainActivity, "Capture failed", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        )
    }

    private suspend fun runInference(bitmap: Bitmap) {
        progressBar.visibility = View.VISIBLE
        previewView.visibility = View.GONE
        resultImageView.visibility = View.GONE
        overlayView.visibility = View.GONE

        try {
            val results = analyzer.processImage(bitmap)
            overlayView.setSegmentationResults(results)
            val annotatedBitmap = drawAnnotations(bitmap, results)

            resultImageView.setImage(ImageSource.bitmap(annotatedBitmap))
            resultImageView.visibility = View.VISIBLE
            overlayView.visibility = View.GONE
            progressBar.visibility = View.GONE
            captureButton.text = "Take Another Photo"
        } catch (e: Exception) {
            Log.e("MainActivity", "Inference failed", e)
            Toast.makeText(this, "Inference failed", Toast.LENGTH_SHORT).show()
            progressBar.visibility = View.GONE
            resetToCamera()
        }
    }

    private fun drawAnnotations(bitmap: Bitmap, results: List<SegmentationResult>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val width = mutableBitmap.width.toFloat()
        val height = mutableBitmap.height.toFloat()

        for (result in results) {
            Log.i("MainActivity", "Drawing annotation for label: ${result.label}, box: ${result.box}")

            // Draw mask with transparency
            val scaledMask = Bitmap.createScaledBitmap(result.mask, width.toInt(), height.toInt(), true)
            val maskPaint = Paint().apply {
                alpha = 80 // Lower alpha for more transparency
                color = getColorForLabel(result.label)
            }
            canvas.drawBitmap(scaledMask, 0f, 0f, maskPaint)

            // Scale and clip bounding box coordinates
            val scaledBox = RectF(
                result.box.left * width,
                result.box.top * height,
                result.box.right * width,
                result.box.bottom * height
            )
            scaledBox.left = scaledBox.left.coerceIn(0f, width)
            scaledBox.top = scaledBox.top.coerceIn(0f, height)
            scaledBox.right = scaledBox.right.coerceIn(0f, width)
            scaledBox.bottom = scaledBox.bottom.coerceIn(0f, height)

            // Draw bounding box
            val boxPaint = Paint().apply {
                color = getColorForLabel(result.label)
                style = Paint.Style.STROKE
                strokeWidth = 5f
            }
            canvas.drawRect(scaledBox, boxPaint)

            // Draw label with background for readability
            val label = result.label
            val textWidth = textPaint.measureText(label)
            val textHeight = textPaint.descent() - textPaint.ascent()
            val textBackgroundPaint = Paint().apply {
                color = Color.BLACK
                style = Paint.Style.FILL
            }
            canvas.drawRect(
                scaledBox.left,
                scaledBox.top - textHeight - 10,
                scaledBox.left + textWidth,
                scaledBox.top - 10,
                textBackgroundPaint
            )
            canvas.drawText(label, scaledBox.left, scaledBox.top - 10, textPaint)
        }
        return mutableBitmap
    }

    private fun getColorForLabel(label: String): Int {
        return when (label) {
            "black_belt" -> Color.BLACK
            "blue_belt" -> Color.BLUE
            "brown_belt" -> Color.parseColor("#8B4513")
            "gold_belt" -> Color.YELLOW
            "gray_belt" -> Color.GRAY
            "green_belt" -> Color.GREEN
            "orange_belt" -> Color.parseColor("#FFA500")
            "purple_belt" -> Color.parseColor("#800080")
            "red_belt" -> Color.RED
            "resistor" -> Color.parseColor("#00CED1") // Turquoise
            "white_belt" -> Color.WHITE
            "yellow_belt" -> Color.parseColor("#FFFF00")
            else -> Color.RED
        }
    }

    private fun resetToCamera() {
        overlayView.setSegmentationResults(emptyList())
        startCamera()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}