package br.ufma.resistorsegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

data class SegmentationResult(val box: RectF, val label: String, val mask: Bitmap)

class SegmentationOverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {
    private var results: List<SegmentationResult> = emptyList()
    private val paint = Paint().apply {
        alpha = 128
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
    }

    fun setSegmentationResults(results: List<SegmentationResult>) {
        this.results = results
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val scaleX = width.toFloat() / 640
        val scaleY = height.toFloat() / 640

        for (result in results) {
            paint.color = getColorForLabel(result.label)
            val scaledBox = RectF(
                result.box.left * scaleX,
                result.box.top * scaleY,
                result.box.right * scaleX,
                result.box.bottom * scaleY
            )
            canvas.drawRect(scaledBox, paint)
            canvas.drawText(
                result.label,
                scaledBox.left,
                scaledBox.top - 10,
                textPaint
            )
        }
    }

    private fun getColorForLabel(label: String): Int {
        return when {
            label.contains("resistor", true) -> Color.RED
            else -> Color.BLUE
        }
    }
}