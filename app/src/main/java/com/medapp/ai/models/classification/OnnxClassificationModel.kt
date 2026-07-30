package com.medapp.ai.models.classification

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import com.medapp.ai.core.ImagePreprocessor
import com.medapp.ai.core.InferenceResult
import com.medapp.ai.core.LabelScore
import com.medapp.ai.core.MedicalModel
import com.medapp.ai.core.ModelDescriptor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.exp
import kotlin.system.measureTimeMillis

/**
 * Generic ONNX Runtime classification model.
 *
 * This class is deliberately generic: it works for DenseNet121 today and
 * for any other image classifier (e.g. a future bone X-ray classifier) as
 * long as it takes one image in and outputs a flat array of class logits.
 * Model-specific details (input size, labels, mean/std) all come from
 * [descriptor], loaded from that model's config.json.
 */
class OnnxClassificationModel(
    override val descriptor: ModelDescriptor
) : MedicalModel {

    private var session: OrtSession? = null
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    override suspend fun load(context: Context) = withContext(Dispatchers.IO) {
        // Models downloaded from GitHub live in app-private local storage
        // (filesDir/models/<id>/). Models still bundled in assets/ (if any)
        // are kept working as a fallback so nothing breaks for those.
        val localFile = com.medapp.ai.remote.ModelDownloadManager.localModelFile(
            context, descriptor.id, descriptor.modelFileName
        )
        val modelBytes = if (localFile.exists()) {
            localFile.readBytes()
        } else {
            val modelPath = "models/${descriptor.id}/${descriptor.modelFileName}"
            context.assets.open(modelPath).use { it.readBytes() }
        }
        val options = OrtSession.SessionOptions()
        session = env.createSession(modelBytes, options)
    }

    override fun unload() {
        session?.close()
        session = null
    }

    override suspend fun runInference(input: Bitmap): InferenceResult = withContext(Dispatchers.Default) {
        val activeSession = session
            ?: return@withContext InferenceResult.Error("النموذج لم يتم تحميله بعد")

        try {
            var scores: List<LabelScore> = emptyList()
            val timeMs = measureTimeMillis {
                // TorchXRayVision normalizes pixel intensities to roughly
                // [-1024, 1024] via: ((gray*2 - 1) * 1024). We reuse the
                // generic (gray - mean) / std formula by solving for the
                // mean/std that produce that exact transform:
                //   target = (gray - 0.5) / (1/2048)  ==  (gray*2 - 1) * 1024
                val tensorData = ImagePreprocessor.bitmapToGrayscaleFloatBuffer(
                    input,
                    descriptor.inputWidth,
                    descriptor.inputHeight,
                    mean = 0.5f,
                    std = 1f / 2048f
                )

                OnnxTensor.createTensor(env, tensorData.buffer, tensorData.shape).use { inputTensor ->
                    val inputName = activeSession.inputNames.iterator().next()
                    val results = activeSession.run(mapOf(inputName to inputTensor))
                    results.use {
                        @Suppress("UNCHECKED_CAST")
                        val rawOutput = (it[0].value as Array<FloatArray>)[0]
                        scores = if (descriptor.multiLabel) {
                            sigmoidToLabelScores(rawOutput, descriptor.labels)
                        } else {
                            softmaxToLabelScores(rawOutput, descriptor.labels)
                        }
                    }
                }
            }

            InferenceResult.Classification(
                predictions = scores.sortedByDescending { it.confidence },
                inferenceTimeMs = timeMs
            )
        } catch (e: Exception) {
            InferenceResult.Error("فشل التحليل: ${e.message}")
        }
    }

    private fun softmaxToLabelScores(logits: FloatArray, labels: List<String>): List<LabelScore> {
        val max = logits.maxOrNull() ?: 0f
        val exps = logits.map { exp((it - max).toDouble()) }
        val sum = exps.sum()
        val probs = exps.map { (it / sum).toFloat() }

        return probs.mapIndexed { i, p ->
            val label = labels.getOrElse(i) { "Class $i" }
            LabelScore(label, p)
        }
    }

    /**
     * Multi-label scoring: each pathology gets its own independent sigmoid
     * probability rather than competing in a single softmax distribution.
     * Required for TorchXRayVision-style models where multiple findings
     * (e.g. Effusion AND Cardiomegaly) can be simultaneously present.
     */
    private fun sigmoidToLabelScores(logits: FloatArray, labels: List<String>): List<LabelScore> {
        return logits.mapIndexed { i, logit ->
            val label = labels.getOrElse(i) { "Class $i" }
            val prob = (1.0 / (1.0 + exp(-logit.toDouble()))).toFloat()
            LabelScore(label, prob)
        }
    }
}
