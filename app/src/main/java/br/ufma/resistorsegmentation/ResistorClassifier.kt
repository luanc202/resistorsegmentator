package br.ufma.resistorsegmentation

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.classifier.ImageClassifier

class ResistorClassifier(context: Context) {
    private var classifier: ImageClassifier? = null

    init {
        try {
            val options = ImageClassifier.ImageClassifierOptions.builder()
                .setMaxResults(5)
                .build()
            classifier = ImageClassifier.createFromFileAndOptions(
                context, "weights/best_float32.tflite", options
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun classify(bitmap: Bitmap): List<Pair<String, Float>> {
        val imageProcessor = ImageProcessor.Builder().build()
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))

        val results = classifier?.classify(tensorImage)
//        Log.i("MainActivity", results.toString())
        return results?.flatMap { it.categories }
            ?.map { it.label to it.score }
            ?.sortedByDescending { it.second }
            ?: emptyList()
    }
}