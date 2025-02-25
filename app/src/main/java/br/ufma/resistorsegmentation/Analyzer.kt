package br.ufma.resistorsegmentation

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import br.ufma.resistorsegmentation.types.SegmentationResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp


class Analyzer(private val overlayView: SegmentationOverlayView, private val context: Context) :
    ImageAnalysis.Analyzer {

    private val MODEL_PATH = "weights/onnx_export_best_float16.tflite"

    private val interpreter: Interpreter by lazy {
        try {
            val modelFile = loadLocalModelFile()

            Interpreter(modelFile, Interpreter.Options().apply {
                addDelegate(GpuDelegate()) // Use GPU for faster inference
            })
            interpreter.allocateTensors()  // Explicitly allocate tensors after initialization
            Log.i("Analyzer", "Interpreter initialized and tensors allocated")
            interpreter
        } catch (e: Exception) {
            Log.e("Analyzer", "Failed to load model", e)
            throw e
        }
    }

    @Throws(IOException::class)
    private fun loadLocalModelFile(): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = context.assets.openFd(MODEL_PATH)
        val inputStream = FileInputStream(fileDescriptor?.fileDescriptor)
        return inputStream.channel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset ?: 0,
            fileDescriptor.declaredLength ?: 0,
        ).also { fileDescriptor.close() }
    }

    override fun analyze(image: ImageProxy) {
        try {
            // Convert ImageProxy to Bitmap
            val bitmap = image.toBitmap()
            val inputTensor = preprocessImage(bitmap)
            Log.i("Analyzer", "Input tensor set")

            val detectionOutput =
                TensorBuffer.createFixedSize(intArrayOf(1, 300, 38), DataType.FLOAT32)
            val maskOutput =
                TensorBuffer.createFixedSize(intArrayOf(1, 160, 160, 32), DataType.FLOAT32)
            val outputs = mapOf(
                0 to detectionOutput.buffer,
                1 to maskOutput.buffer
            )

            Log.i("Analyzer", "Running inference")
            // Run inference
            interpreter.runForMultipleInputsOutputs(arrayOf(inputTensor.buffer), outputs)

            Log.i("Analyzer", "Postprocessing output")
            // Postprocess the outputs
            val detectionArray =
                Array(300) { i -> FloatArray(38) { j -> detectionOutput.floatArray[i * 38 + j] } }
            val maskArray =
                Array(160) { i -> Array(160) { j -> FloatArray(32) { k -> maskOutput.floatArray[(i * 160 + j) * 32 + k] } } }
            val segmentationResults = postprocessOutput(detectionArray, maskArray)
            Log.i("Analyzer", "Postprocessing done")
            // Update overlay view on UI thread
            overlayView.post {
                overlayView.setSegmentationResults(segmentationResults)
                overlayView.invalidate()
            }

            Log.i("Analyzer", "Overlay view updated")
        } catch (e: Exception) {
            Log.e("Analyzer", "Inference failed", e)
        } finally {
            image.close()
        }
    }

    private fun preprocessImage(bitmap: Bitmap): TensorBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        val tensorBuffer =
            TensorBuffer.createFixedSize(intArrayOf(1, 640, 640, 3), DataType.FLOAT32)
        val pixels = IntArray(640 * 640)
        resized.getPixels(pixels, 0, 640, 0, 0, 640, 640)
        val floatBuffer = tensorBuffer.buffer.asFloatBuffer()
        for (i in pixels.indices) {
            floatBuffer.put(((pixels[i] shr 16 and 0xFF) / 255f)) // R
            floatBuffer.put(((pixels[i] shr 8 and 0xFF) / 255f))  // G
            floatBuffer.put(((pixels[i] and 0xFF) / 255f))        // B
        }
        return tensorBuffer
    }

    private fun postprocessOutput(
        detectionOutput: Array<FloatArray>,  // Shape [300, 38]
        maskOutput: Array<Array<FloatArray>> // Shape [32, 160, 160]
    ): List<SegmentationResult> {
        // Constants (adjust based on your model)
        val numClasses = 1  // Adjust based on your dataset (e.g., 1 if 33rd value is a class score)
        val numCoefficients =
            32  // Assuming 32 mask coefficients (38 - 4 box - 1 objectness - 1 class = 32)
        val stride = 640 / 160  // Downscale factor (4)
        val confidenceThreshold = 0.25f
        val nmsThreshold = 0.45f

        // Step 1: Parse detection output
        val detections = mutableListOf<Triple<RectF, Float, Int>>()  // (box, score, class)
        val maskCoefficients = mutableListOf<FloatArray>()  // Coefficients for each detection

        for (i in 0 until 300) {  // Iterate over 300 detections
            val xCenter = detectionOutput[i][0]  // Column 0: x_center
            val yCenter = detectionOutput[i][1]  // Column 1: y_center
            val width = detectionOutput[i][2]    // Column 2: width
            val height = detectionOutput[i][3]   // Column 3: height
            val objectness = detectionOutput[i][4]  // Column 4: objectness score

            // Check confidence
            if (objectness < confidenceThreshold) continue

            // Find max class score (assuming 1 class for simplicity, adjust as needed)
            val classScore = detectionOutput[i][5]  // Column 5: class score (e.g., single class)
            val totalScore = objectness * classScore
            if (totalScore < confidenceThreshold) continue

            // Calculate bounding box
            val left = xCenter - width / 2f
            val top = yCenter - height / 2f
            val right = xCenter + width / 2f
            val bottom = yCenter + height / 2f
            val box = RectF(left, top, right, bottom)

            // Store detection
            detections.add(Triple(box, totalScore, 0))  // Class ID 0 if single class

            // Extract mask coefficients (columns 6 to 37, assuming 32 coefficients)
            val coefficients = FloatArray(numCoefficients) { j ->
                detectionOutput[i][6 + j]
            }
            maskCoefficients.add(coefficients)
        }

        // Step 2: Apply NMS
        val selectedIndices = applyNMS(detections, nmsThreshold)
        val results = mutableListOf<SegmentationResult>()

        // Step 3: Generate segmentation masks
        for (idx in selectedIndices) {
            val (box, _, classId) = detections[idx]
            val coefficients = maskCoefficients[idx]

            // Compute mask by combining coefficients with prototypes
            val mask = FloatArray(160 * 160)
            for (y in 0 until 160) {
                for (x in 0 until 160) {
                    var sum = 0f
                    for (p in 0 until 32) {
                        sum += coefficients[p] * maskOutput[p][y][x]  // Channel-first indexing
                    }
                    mask[y * 160 + x] = sigmoid(sum)  // Apply sigmoid to get [0, 1]
                }
            }

            // Convert mask to Bitmap (threshold at 0.5)
            val maskBitmap = Bitmap.createBitmap(160, 160, Bitmap.Config.ARGB_8888)
            for (y in 0 until 160) {
                for (x in 0 until 160) {
                    val value = if (mask[y * 160 + x] > 0.5f) 255 else 0
                    maskBitmap.setPixel(x, y, Color.argb(value, 255, 255, 255))
                }
            }

            // Scale mask to input size (640x640)
            val scaledMask = Bitmap.createScaledBitmap(maskBitmap, 640, 640, true)

            // Map class ID to label (adjust as needed)
            val label = "class_$classId"  // Replace with actual class names

            results.add(SegmentationResult(box, label, scaledMask))
        }

        return results
    }

    // Helper: Apply Non-Maximum Suppression
    private fun applyNMS(detections: List<Triple<RectF, Float, Int>>, threshold: Float): List<Int> {
        val sorted = detections.mapIndexed { index, triple -> index to triple.second }
            .sortedByDescending { it.second }
        val selected = mutableListOf<Int>()
        val suppressed = BooleanArray(detections.size)

        for (i in sorted.indices) {
            val idx = sorted[i].first
            if (suppressed[idx]) continue
            selected.add(idx)
            val box1 = detections[idx].first

            for (j in i + 1 until sorted.size) {
                val idx2 = sorted[j].first
                if (suppressed[idx2]) continue
                val box2 = detections[idx2].first
                if (computeIoU(box1, box2) > threshold) {
                    suppressed[idx2] = true
                }
            }
        }
        return selected
    }

    // Helper: Compute Intersection over Union (IoU)
    private fun computeIoU(box1: RectF, box2: RectF): Float {
        val x1 = maxOf(box1.left, box2.left)
        val y1 = maxOf(box1.top, box2.top)
        val x2 = minOf(box1.right, box2.right)
        val y2 = minOf(box1.bottom, box2.bottom)
        val intersection = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val area1 = (box1.right - box1.left) * (box1.bottom - box1.top)
        val area2 = (box2.right - box2.left) * (box2.bottom - box2.top)
        return intersection / (area1 + area2 - intersection)
    }

    // Helper: Sigmoid function
    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))
}