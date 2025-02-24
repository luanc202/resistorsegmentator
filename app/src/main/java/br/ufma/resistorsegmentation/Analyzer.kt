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
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp


class Analyzer(private val overlayView: SegmentationOverlayView, private val context: Context) :
    ImageAnalysis.Analyzer {

    private val MODEL_PATH = "weights/best_float32.tflite"

    private val interpreter: Interpreter by lazy {
        try {
            val modelFile = loadLocalModelFile()

            Interpreter(modelFile)
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
        )
    }

    override fun analyze(image: ImageProxy) {
        try {
            // Convert ImageProxy to Bitmap
            val bitmap = image.toBitmap()
            val inputTensor = preprocessImage(bitmap)
            Log.i("Analyzer", "Input tensor set")

            val detectionOutput = Array(1) { Array(45) { FloatArray(8400) } }  // [1, 45, 8400]
            val maskOutput = Array(1) { FloatArray(160 * 160 * 32) }  // [1, 160, 160, 32] flattened
            val outputs = mapOf(
                0 to detectionOutput,  // Output 0: detections
                1 to maskOutput        // Output 1: mask prototypes
            )

            Log.i("Analyzer", "Running inference")

            // Run inference
            interpreter.runForMultipleInputsOutputs(arrayOf(inputTensor.buffer), outputs)

            Log.i("Analyzer", "Postprocessing output")
            // Postprocess the outputs
            val segmentationResults = postprocessOutput(detectionOutput[0], maskOutput[0])
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
        detectionOutput: Array<FloatArray>,  // Shape [45, 8400]
        maskOutput: FloatArray              // Shape [160, 160, 32]
    ): List<SegmentationResult> {
        Log.i("Analyzer", "Detection output shape: [${detectionOutput.size}, ${detectionOutput[0].size}, ${detectionOutput[0][0]}]")
        // Constants (adjust based on your model)
        val numClasses = 80  // e.g., COCO dataset has 80 classes
        val numCoefficients = 32  // Number of mask coefficients per detection
        val stride = 640 / 160  // Downscale factor (4 for 640x640 input, 160x160 masks)
        val confidenceThreshold = 0.25f
        val nmsThreshold = 0.45f

        // Step 1: Parse detection output
        val detections = mutableListOf<Triple<RectF, Float, Int>>()  // (box, score, class)
        val maskCoefficients = mutableListOf<FloatArray>()  // Coefficients for each detection

        for (i in 0 until 8400) {  // Iterate over 8400 grid points
            val xCenter = detectionOutput[0][i]  // Row 0: x_center
            val yCenter = detectionOutput[1][i]  // Row 1: y_center
            val width = detectionOutput[2][i]    // Row 2: width
            val height = detectionOutput[3][i]   // Row 3: height
            val objectness = detectionOutput[4][i]  // Row 4: objectness score

            // Check confidence
            if (objectness < confidenceThreshold) continue

            // Find max class score
            var maxScore = 0f
            var maxClass = -1
            for (c in 0 until numClasses) {
                val score = detectionOutput[5 + c][i]  // Rows 5 to 84: class scores
                if (score > maxScore) {
                    maxScore = score
                    maxClass = c
                }
            }
            val totalScore = objectness * maxScore
            if (totalScore < confidenceThreshold) continue

            // Calculate bounding box
            val left = xCenter - width / 2f
            val top = yCenter - height / 2f
            val right = xCenter + width / 2f
            val bottom = yCenter + height / 2f
            val box = RectF(left, top, right, bottom)

            // Store detection
            detections.add(Triple(box, totalScore, maxClass))

            // Extract mask coefficients (rows 85 to 116, assuming 80 classes + 5)
            val coefficients = FloatArray(numCoefficients) { j ->
                detectionOutput[5 + numClasses + j][i]
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
                        sum += coefficients[p] * maskOutput[(y * 160 + x) * 32 + p]
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

            // Map class ID to label (example mapping, adjust as needed)
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