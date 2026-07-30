package com.medapp.ai.core

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Shared image -> tensor conversion helpers.
 *
 * Every classification-style model needs roughly the same steps (resize,
 * normalize, arrange channels). Keeping this in one place means new models
 * usually only need to supply their own mean/std values, not rewrite this logic.
 */
object ImagePreprocessor {

    /**
     * Resize [bitmap] to [width]x[height] and convert to an NCHW float tensor
     * normalized with the given per-channel mean/std (ImageNet defaults are
     * the common case for models pretrained via torchvision/MONAI).
     */
    fun bitmapToNchwFloatBuffer(
        bitmap: Bitmap,
        width: Int,
        height: Int,
        mean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f),
        std: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f)
    ): FloatBuffer3D {
        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val pixels = IntArray(width * height)
        resized.getPixels(pixels, 0, width, 0, 0, width, height)

        val buffer = ByteBuffer
            .allocateDirect(3 * width * height * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        // NCHW layout: channel planes, not interleaved
        val channelPlanes = Array(3) { FloatArray(width * height) }
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            channelPlanes[0][i] = (r - mean[0]) / std[0]
            channelPlanes[1][i] = (g - mean[1]) / std[1]
            channelPlanes[2][i] = (b - mean[2]) / std[2]
        }
        channelPlanes.forEach { buffer.put(it) }
        buffer.rewind()

        if (resized !== bitmap) resized.recycle()

        return FloatBuffer3D(buffer, longArrayOf(1, 3, height.toLong(), width.toLong()))
    }

    /** Convert grayscale (single-channel) images, common for X-ray. */
    fun bitmapToGrayscaleFloatBuffer(
        bitmap: Bitmap,
        width: Int,
        height: Int,
        mean: Float = 0.5f,
        std: Float = 0.5f
    ): FloatBuffer3D {
        val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val pixels = IntArray(width * height)
        resized.getPixels(pixels, 0, width, 0, 0, width, height)

        val buffer = ByteBuffer
            .allocateDirect(width * height * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

        for (p in pixels) {
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            val gray = 0.299f * r + 0.587f * g + 0.114f * b
            buffer.put((gray - mean) / std)
        }
        buffer.rewind()

        if (resized !== bitmap) resized.recycle()

        return FloatBuffer3D(buffer, longArrayOf(1, 1, height.toLong(), width.toLong()))
    }
}

/** A float tensor buffer plus its shape, ready to hand to ONNX Runtime's OnnxTensor.createTensor. */
data class FloatBuffer3D(val buffer: java.nio.FloatBuffer, val shape: LongArray)
