package com.medapp.ai.core

import android.content.Context
import android.graphics.Bitmap

/**
 * The single contract every model in the app must implement.
 *
 * This is the extensibility seam: to add a new model (segmentation, a new
 * classifier, etc.) you write ONE class that implements this interface and
 * register it — see [com.medapp.ai.registry.ModelRegistry]. Nothing in the
 * UI layer needs to change.
 */
interface MedicalModel {

    /** Static metadata describing this model (used to render UI, validate inputs, etc). */
    val descriptor: ModelDescriptor

    /**
     * Load the model into memory (e.g. create the ONNX Runtime session).
     * Must be safe to call once before any [runInference] call.
     */
    suspend fun load(context: Context)

    /** Free any native resources. Call when the model is no longer needed. */
    fun unload()

    /**
     * Run the full pipeline: preprocess -> inference -> postprocess.
     * Implementations handle their own tensor shape / normalization concerns
     * internally, so callers only ever deal in [Bitmap] in and [InferenceResult] out.
     */
    suspend fun runInference(input: Bitmap): InferenceResult
}

/**
 * Metadata describing a model, primarily driven by each model's config.json
 * under assets/models/<model_id>/. Kept separate from the model logic so the
 * UI (model list, capability badges, etc.) can be built generically.
 */
data class ModelDescriptor(
    val id: String,
    val displayNameAr: String,
    val displayNameEn: String,
    val modality: Modality,
    val task: ModelTask,
    val descriptionAr: String,
    val inputWidth: Int,
    val inputHeight: Int,
    val fileSizeMb: Int,
    val labels: List<String> = emptyList(),
    val modelFileName: String = "model.onnx",
    val multiLabel: Boolean = false,
    /** Whether this model's weight file is ready to use locally, or still needs downloading. */
    val downloadStatus: ModelDownloadStatus = ModelDownloadStatus.BUNDLED
)

enum class ModelDownloadStatus {
    /** Shipped inside the app's assets — always ready, nothing to download. */
    BUNDLED,
    /** Listed on GitHub but not yet downloaded to this device. */
    AVAILABLE_REMOTE,
    /** Downloaded from GitHub and ready to use fully offline. */
    DOWNLOADED
}

enum class Modality {
    XRAY, CT, MRI, PATHOLOGY, OTHER
}

enum class ModelTask {
    CLASSIFICATION, SEGMENTATION, ANOMALY_DETECTION
}

/**
 * Generic result wrapper. Only one of the fields is populated depending on
 * [ModelTask] — UI code checks `task` on the descriptor to know which to render.
 */
sealed class InferenceResult {

    data class Classification(
        val predictions: List<LabelScore>,
        val inferenceTimeMs: Long
    ) : InferenceResult()

    data class Segmentation(
        val maskBitmap: Bitmap,
        val affectedAreaPercent: Float,
        val inferenceTimeMs: Long
    ) : InferenceResult()

    data class Error(val message: String) : InferenceResult()
}

data class LabelScore(val label: String, val confidence: Float)
