package br.ufma.resistorsegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import br.ufma.resistorsegmentation.types.SegmentationResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp


class Analyzer(private val context: Context) {

    private companion object {
        const val MODEL_PATH = "weights/best_float32.tflite"

        // Input shape [1, channels, height, width]
        const val INPUT_BATCH_SIZE = 1
        const val INPUT_CHANNELS = 3
        const val INPUT_HEIGHT = 640
        const val INPUT_WIDTH = 640

        // Detection output shape [1, num_detections, detection_values]
        const val DETECTION_BATCH_SIZE = 1
        const val NUM_DETECTIONS = 300  // Updated from 38
        const val DETECTION_VALUES = 38  // Updated from 300

        // Mask output shape [1, mask_height, mask_width, num_coefficients]
        const val MASK_BATCH_SIZE = 1
        const val MASK_HEIGHT = 160
        const val MASK_WIDTH = 160
        const val MASK_COEFFICIENTS = 32
    }

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
                channel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
                    .also { fileDescriptor.close() }
            }
        }
    }

    private fun preprocessImage(bitmap: Bitmap): TensorBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true)
        val tensorBuffer = TensorBuffer.createFixedSize(
            intArrayOf(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
            DataType.FLOAT32
        )
        val floatBuffer = tensorBuffer.buffer.asFloatBuffer()
        val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
        resized.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)
        val numElements = INPUT_WIDTH * INPUT_HEIGHT
        for (i in 0 until numElements) {
            val pixel = pixels[i]
            floatBuffer.put(i, ((pixel shr 16 and 0xFF) / 255f)) // R
            floatBuffer.put(i + numElements, ((pixel shr 8 and 0xFF) / 255f)) // G
            floatBuffer.put(i + 2 * numElements, ((pixel and 0xFF) / 255f)) // B
        }
        return tensorBuffer
    }

    suspend fun processImage(bitmap: Bitmap): List<SegmentationResult> {
        return withContext(Dispatchers.Default) {
            try {
                val startTime = System.currentTimeMillis()
                val inputTensor = preprocessImage(bitmap)
                Log.i("Analyzer", "Input tensor set in ${System.currentTimeMillis() - startTime}ms")

                val detectionOutput = TensorBuffer.createFixedSize(
                    intArrayOf(DETECTION_BATCH_SIZE, NUM_DETECTIONS, DETECTION_VALUES),
                    DataType.FLOAT32
                )
                val maskOutput = TensorBuffer.createFixedSize(
                    intArrayOf(MASK_BATCH_SIZE, MASK_HEIGHT, MASK_WIDTH, MASK_COEFFICIENTS),
                    DataType.FLOAT32
                )
                val outputs = mapOf(0 to detectionOutput.buffer, 1 to maskOutput.buffer)

                val inferenceStart = System.currentTimeMillis()
                Log.i("Analyzer", "Running inference")
                interpreter.runForMultipleInputsOutputs(arrayOf(inputTensor.buffer), outputs)
                Log.i("Analyzer", "Inference done in ${System.currentTimeMillis() - inferenceStart}ms")
                Log.i("Analyzer", "maskOutput size: ${maskOutput.floatArray.size}")
                Log.i("Analyzer", "Raw maskOutput sample: ${maskOutput.floatArray.take(10).joinToString()}")

                Log.i("Analyzer", "Postprocessing output")
                val detectionStart = System.currentTimeMillis()
                val detectionArray = Array(NUM_DETECTIONS) { i ->
                    FloatArray(DETECTION_VALUES) { j -> detectionOutput.floatArray[i * DETECTION_VALUES + j] }
                }
                Log.i("Analyzer", "detectionArray set in ${System.currentTimeMillis() - detectionStart}ms")
                Log.i("Analyzer", "detectionArray sample: ${detectionArray[0].joinToString()}")

                val maskStart = System.currentTimeMillis()
                val maskFlatArray = withContext(Dispatchers.Default) {
                    FloatArray(MASK_HEIGHT * MASK_WIDTH * MASK_COEFFICIENTS).apply {
                        maskOutput.floatArray.copyInto(this)
                    }
                }
                Log.i("Analyzer", "maskArray set in ${System.currentTimeMillis() - maskStart}ms")
                Log.i("Analyzer", "maskFlatArray sample: ${maskFlatArray.take(10).joinToString()}")

                val postStart = System.currentTimeMillis()
                val results = postprocessOutput(detectionArray, maskFlatArray)
                Log.i("Analyzer", "Postprocessing complete in ${System.currentTimeMillis() - postStart}ms")
                results
            } catch (e: Exception) {
                Log.e("Analyzer", "Inference failed", e)
                emptyList()
            }
        }
    }

    private fun postprocessOutput(
        detectionOutput: Array<FloatArray>,
        maskFlatArray: FloatArray
    ): List<SegmentationResult> {
        val numClasses = 12  // Updated to 12 from metadata
        val classNames = listOf(
            "black_belt", "blue_belt", "brown_belt", "gold_belt", "gray_belt",
            "green_belt", "orange_belt", "purple_belt", "red_belt", "resistor",
            "white_belt", "yellow_belt"
        )
        val numCoefficients = 21  // 38 - 4 (box) - 1 (objectness) - 12 (classes) = 21
        val stride = 32
        val confidenceThreshold = 0.25f
        val nmsThreshold = 0.45f

        Log.i("Analyzer", "Raw detection sample (first entry): ${detectionOutput[0].joinToString()}")

        val detections = mutableListOf<Triple<RectF, Float, Int>>()
        val maskCoefficients = mutableListOf<FloatArray>()

        for (i in 0 until NUM_DETECTIONS) {
            val xCenter = detectionOutput[i][0]
            val yCenter = detectionOutput[i][1]
            val width = detectionOutput[i][2]
            val height = detectionOutput[i][3]
            val objectness = detectionOutput[i][4]

            if (objectness < confidenceThreshold) continue

            var maxScore = 0f
            var maxClass = -1
            for (c in 0 until numClasses) {
                val score = detectionOutput[i][5 + c]
                if (score > maxScore) {
                    maxScore = score
                    maxClass = c
                }
            }
            val totalScore = objectness * maxScore
            if (totalScore < confidenceThreshold) continue

            val left = xCenter - width / 2f
            val top = yCenter - height / 2f
            val right = xCenter + width / 2f
            val bottom = yCenter + height / 2f
            val box = RectF(left, top, right, bottom)

            detections.add(Triple(box, totalScore, maxClass))

            val coefficients = FloatArray(numCoefficients) { j -> detectionOutput[i][5 + numClasses + j] }
            maskCoefficients.add(coefficients)
        }

        Log.i("Analyzer", "Detections before NMS: ${detections.size}")

        val selectedIndices = applyNMS(detections, nmsThreshold)
        val results = mutableListOf<SegmentationResult>()

        for (idx in selectedIndices) {
            val (box, score, classId) = detections[idx]
            val coefficients = maskCoefficients[idx]

            val mask = FloatArray(MASK_HEIGHT * MASK_WIDTH)
            for (y in 0 until MASK_HEIGHT) {
                for (x in 0 until MASK_WIDTH) {
                    var sum = 0f
                    for (p in 0 until numCoefficients) {
                        val index = (y * MASK_WIDTH + x) * MASK_COEFFICIENTS + p
                        sum += coefficients[p] * maskFlatArray[index]
                    }
                    mask[y * MASK_WIDTH + x] = sigmoid(sum)
                }
            }
            Log.i("Analyzer", "Mask sample for ${classNames[classId]}: ${mask.take(10).joinToString()}")

            val maskBitmap = Bitmap.createBitmap(MASK_WIDTH, MASK_HEIGHT, Bitmap.Config.ARGB_8888)
            for (y in 0 until MASK_HEIGHT) {
                for (x in 0 until MASK_WIDTH) {
                    val value = if (mask[y * MASK_WIDTH + x] > 0.1f) 255 else 0 // Lowered threshold
                    maskBitmap.setPixel(x, y, Color.argb(255, value, value, value))
                }
            }

            val scaledMask = Bitmap.createScaledBitmap(maskBitmap, INPUT_WIDTH, INPUT_HEIGHT, true)
            val label = classNames[classId]
            results.add(SegmentationResult(box, label, scaledMask))

            Log.i("Analyzer", "Detection: Label=$label, Score=$score, Box=[${box.left}, ${box.top}, ${box.right}, ${box.bottom}]")
        }

        Log.i("Analyzer", "Final detections after NMS: ${results.size}")
        return results
    }

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

    private fun computeIoU(box1: RectF, box2: RectF): Float {
        val x1 = maxOf(box1.left, box2.left)
        val y1 = maxOf(box1.top, box2.top)
        val x2 = minOf(box1.right, box2.right)
        val y2 = minOf(box1.bottom, box2.bottom)
        val intersection = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val area1 = (box1.right - box1.left) * (box1.bottom - box1.top)
        val area2 = (box2.right - box2.left) * (box2.bottom - box2.top)
        return if (area1 + area2 - intersection == 0f) 0f else intersection / (area1 + area2 - intersection)
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))
}