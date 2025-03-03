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
    private val paint = Paint().apply { alpha = 128 }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
    }

    fun setSegmentationResults(results: List<SegmentationResult>) {
        this.results = results
        invalidate() // Trigger redraw
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scaleX = width.toFloat() / 640f
        val scaleY = height.toFloat() / 640f

        Log.i("SegmentationOverlayView", "Drawing ${results.size} results")

        for (result in results) {
            // Log mask sample to debug
            val maskPixels = IntArray(10 * 10)
            result.mask.getPixels(maskPixels, 0, 10, 0, 0, 10, 10)
            Log.i("SegmentationOverlayView", "Mask sample (top-left 10x10): ${maskPixels.take(10).joinToString()}")

            val scaledMask = Bitmap.createScaledBitmap(result.mask, width, height, true)
            paint.color = getColorForLabel(result.label)
            canvas.drawBitmap(scaledMask, 0f, 0f, paint)

            val scaledBox = RectF(
                result.box.left * scaleX,
                result.box.top * scaleY,
                result.box.right * scaleX,
                result.box.bottom * scaleY
            )
            canvas.drawText(result.label, scaledBox.left, scaledBox.top - 10, textPaint)
        }
    }

    private fun getColorForLabel(label: String): Int {
        return when (label) {
            "black_belt" -> Color.BLACK
            "blue_belt" -> Color.BLUE
            "brown_belt" -> Color.parseColor("#8B4513")
            "gold_belt" -> Color.YELLOW
            "gray_belt" -> Color.GRAY
            "orange_belt" -> Color.parseColor("#FFA500")
            "red_belt" -> Color.RED
            "resistor" -> Color.GREEN
            "yellow_belt" -> Color.YELLOW
            else -> Color.RED
        }
    }
}