package br.ufma.resistorsegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.View
import br.ufma.resistorsegmentation.types.SegmentationResult

class SegmentationOverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private var results: List<SegmentationResult> = emptyList()
    private val paint = Paint().apply { alpha = 128 } // Semi-transparent masks
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
    }

    fun setSegmentationResults(results: List<SegmentationResult>) {
        this.results = results
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scaleX = width.toFloat() / 640 // Assuming analysis size is 640x640
        val scaleY = height.toFloat() / 640

        Log.i("SegmentationOverlayView", "Initializing drawing of results")

        for (result in results) {

            // Scale mask to display size
            val scaledMask = Bitmap.createScaledBitmap(result.mask, width, height, true)
            paint.color = getColorForLabel(result.label) // Define color mapping
            canvas.drawBitmap(scaledMask, 0f, 0f, paint)

            // Scale bounding box
            val scaledBox = RectF(
                result.box.left * scaleX,
                result.box.top * scaleY,
                result.box.right * scaleX,
                result.box.bottom * scaleY
            )
            canvas.drawText(result.label, scaledBox.left, scaledBox.top - 10, textPaint)
            Log.i("SegmentationOverlayView", "Finished drawing of results")
        }
    }

    private fun getColorForLabel(label: String): Int {
        // Map labels to colors, e.g., "person" -> Color.RED, "car" -> Color.BLUE
        return Color.RED // Placeholder
    }
}