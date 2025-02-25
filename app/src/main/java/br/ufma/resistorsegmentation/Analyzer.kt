package br.ufma.resistorsegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import br.ufma.resistorsegmentation.types.SegmentationResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp


class Analyzer(private val overlayView: SegmentationOverlayView, private val context: Context) {

    private val MODEL_PATH = "weights/nmsbest_float32.tflite" // Update to your model path

    private val interpreter: Interpreter by lazy {
        try {
            val modelFile = loadLocalModelFile()
            val interpreter = Interpreter(modelFile, Interpreter.Options().apply {
                setNumThreads(4)
            })
            interpreter.allocateTensors()
            Log.i("Analyzer", "Interpreter initialized and tensors allocated")
            interpreter
        } catch (e: Exception) {
            Log.e("Analyzer", "Failed to load model", e)
            throw e
        }
    }

    @Throws(IOException::class)
    private fun loadLocalModelFile(): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(MODEL_PATH)
        return FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
            inputStream.channel.use { channel ->
                channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.startOffset,
                    fileDescriptor.declaredLength
                ).also {
                    fileDescriptor.close()
                }
            }
        }
    }

    fun processImage(bitmap: Bitmap): List<SegmentationResult> {
        try {
            val inputTensor = preprocessImage(bitmap)
            Log.i("Analyzer", "Input tensor set")

            val detectionOutput = TensorBuffer.createFixedSize(intArrayOf(1, 300, 38), DataType.FLOAT32)
            val maskOutput = TensorBuffer.createFixedSize(intArrayOf(1, 160, 160, 32), DataType.FLOAT32)
            val outputs = mapOf(
                0 to detectionOutput.buffer,
                1 to maskOutput.buffer
            )

            Log.i("Analyzer", "Running inference")
            interpreter.runForMultipleInputsOutputs(arrayOf(inputTensor.buffer), outputs)

            Log.i("Analyzer", "Postprocessing output")
            val detectionArray = Array(300) { i -> FloatArray(38) { j -> detectionOutput.floatArray[i * 38 + j] } }
            val maskArray = Array(160) { i -> Array(160) { j -> FloatArray(32) { k -> maskOutput.floatArray[(i * 160 + j) * 32 + k] } } }
            return postprocessOutput(detectionArray, maskArray)
        } catch (e: Exception) {
            Log.e("Analyzer", "Inference failed", e)
            throw e
        }
    }

    private fun preprocessImage(bitmap: Bitmap): TensorBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        val tensorBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 3, 640, 640), DataType.FLOAT32)
        val floatBuffer = tensorBuffer.buffer.asFloatBuffer()
        val pixels = IntArray(640 * 640)
        resized.getPixels(pixels, 0, 640, 0, 0, 640, 640)
        val numElements = 640 * 640
        for (i in 0 until numElements) {
            val pixel = pixels[i]
            floatBuffer.put(i, ((pixel shr 16 and 0xFF) / 255f))          // R channel
            floatBuffer.put(i + numElements, ((pixel shr 8 and 0xFF) / 255f))  // G channel
            floatBuffer.put(i + 2 * numElements, ((pixel and 0xFF) / 255f))    // B channel
        }
        return tensorBuffer
    }

    private fun postprocessOutput(
        detectionOutput: Array<FloatArray>,  // Shape [300, 38]
        maskOutput: Array<Array<FloatArray>> // Shape [160, 160, 32]
    ): List<SegmentationResult> {
        // Your existing postprocessOutput logic remains unchanged...
        val numClasses = 1
        val numCoefficients = 32
        val stride = 640 / 160
        val confidenceThreshold = 0.25f
        val nmsThreshold = 0.45f

        val detections = mutableListOf<Triple<RectF, Float, Int>>()
        val maskCoefficients = mutableListOf<FloatArray>()

        for (i in 0 until 300) {
            val xCenter = detectionOutput[i][0]
            val yCenter = detectionOutput[i][1]
            val width = detectionOutput[i][2]
            val height = detectionOutput[i][3]
            val objectness = detectionOutput[i][4]

            if (objectness < confidenceThreshold) continue

            val classScore = detectionOutput[i][5]
            val totalScore = objectness * classScore
            if (totalScore < confidenceThreshold) continue

            val left = xCenter - width / 2f
            val top = yCenter - height / 2f
            val right = xCenter + width / 2f
            val bottom = yCenter + height / 2f
            val box = RectF(left, top, right, bottom)

            detections.add(Triple(box, totalScore, 0))

            val coefficients = FloatArray(numCoefficients) { j -> detectionOutput[i][6 + j] }
            maskCoefficients.add(coefficients)
        }

        val selectedIndices = applyNMS(detections, nmsThreshold)
        val results = mutableListOf<SegmentationResult>()

        for (idx in selectedIndices) {
            val (box, _, classId) = detections[idx]
            val coefficients = maskCoefficients[idx]

            val mask = FloatArray(160 * 160)
            for (y in 0 until 160) {
                for (x in 0 until 160) {
                    var sum = 0f
                    for (p in 0 until 32) {
                        sum += coefficients[p] * maskOutput[y][x][p]
                    }
                    mask[y * 160 + x] = sigmoid(sum)
                }
            }

            val maskBitmap = Bitmap.createBitmap(160, 160, Bitmap.Config.ARGB_8888)
            for (y in 0 until 160) {
                for (x in 0 until 160) {
                    val value = if (mask[y * 160 + x] > 0.5f) 255 else 0
                    maskBitmap.setPixel(x, y, Color.argb(value, 255, 255, 255))
                }
            }

            val scaledMask = Bitmap.createScaledBitmap(maskBitmap, 640, 640, true)
            val label = "class_$classId"
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