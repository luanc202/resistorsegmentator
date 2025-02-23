package br.ufma.resistorsegmentation.types

import android.graphics.Bitmap
import android.graphics.RectF

data class SegmentationResult(val box: RectF, val label: String, val mask: Bitmap)
