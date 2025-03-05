package br.ufma.resistorsegmentation

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import android.graphics.BitmapFactory
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage

class YuvToRgbConverter(private val context: Unit) {
    fun yuvToRgb(image: ImageProxy, output: Bitmap) {
        val yuvBytes = yuvToByteArray(image)
        val yuvImage = YuvImage(yuvBytes, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val jpegBytes = out.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
        output.setPixels(
            IntArray(bitmap.width * bitmap.height).apply {
                bitmap.getPixels(this, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            },
            0, bitmap.width, 0, 0, bitmap.width, bitmap.height
        )
    }

    private fun yuvToByteArray(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        return nv21
    }
}

@OptIn(ExperimentalGetImage::class)
fun ImageProxy.toBitmap(): Bitmap {
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    YuvToRgbConverter(this.image?.let { android.media.ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 2).close() }
        ?: throw IllegalStateException("Context not available")).yuvToRgb(this, bitmap)
    return bitmap
}