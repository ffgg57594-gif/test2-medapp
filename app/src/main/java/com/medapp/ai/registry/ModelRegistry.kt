package com.medapp.ai.registry

import android.content.Context
import com.medapp.ai.core.MedicalModel
import com.medapp.ai.core.ModelDescriptor
import com.medapp.ai.core.ModelDownloadStatus
import com.medapp.ai.core.Modality
import com.medapp.ai.core.ModelTask
import com.medapp.ai.models.classification.OnnxClassificationModel
import com.medapp.ai.remote.GitHubModelSource
import com.medapp.ai.remote.ModelDownloadManager
import org.json.JSONObject

/**
 * Central place that knows about every model available in the app.
 *
 * The app itself ships with (usually) zero or very few models bundled in
 * assets/. Instead, the list of available models is fetched from the
 * `models/` folder of the project's GitHub repo — anyone can publish a new
 * model there and it shows up here automatically, without an app update.
 * The user explicitly taps a model to download it before it can run
 * (inference always happens locally afterward, never over the network).
 *
 * HOW TO PUBLISH A NEW MODEL (no app code changes needed):
 *   1. Push a new folder to the repo: models/<your_model_id>/
 *        - config.json  (same schema as existing models)
 *        - model.onnx   (or whatever model_file_name points to)
 *   That's it — it will appear in the app's model list next refresh.
 *
 * (Optional legacy path: a model can still be bundled directly into
 * app/src/main/assets/models/<id>/ at build time — e.g. for a default
 * always-available model — by adding its id to [BUNDLED_MODEL_IDS].)
 */
object ModelRegistry {

    /** Models shipped inside the APK itself (assets/models/). Usually empty. */
    private val BUNDLED_MODEL_IDS = listOf<String>()

    private val descriptorCache = mutableMapOf<String, ModelDescriptor>()

    /**
     * Fetch the full list of models to show in the model-picker screen.
     *
     * Always includes: bundled models + anything already downloaded to this
     * device (both read from local storage only, no network required — this
     * is what keeps previously-downloaded models usable while offline).
     *
     * If a network connection is available, this list is then merged with
     * whatever's currently published on GitHub, so newly-published models
     * (not yet downloaded) also show up with a "download" option. If the
     * network call fails (no internet), the local-only list is still
     * returned — the screen never goes empty just because you're offline.
     */
    fun listAvailable(context: Context): List<ModelDescriptor> {
        val bundled = BUNDLED_MODEL_IDS.mapNotNull { id ->
            runCatching { loadBundledDescriptor(context, id) }.getOrNull()
        }

        val downloaded = ModelDownloadManager.listDownloadedModelIds(context)
            .mapNotNull { modelId ->
                runCatching { loadDownloadedDescriptor(context, modelId) }.getOrNull()
            }

        val remote = runCatching { GitHubModelSource.fetchAvailableModels() }
            .getOrDefault(emptyList())
            .mapNotNull { remoteModel ->
                // Each model is parsed independently: if one model's
                // config.json is malformed (e.g. an invalid "modality"
                // value), skip just that one instead of breaking the whole
                // list for every other model.
                runCatching { toDescriptor(context, remoteModel) }.getOrNull()
            }

        // Merge, preferring the already-downloaded version of a model over
        // its remote listing (avoids duplicate cards, and reflects reality
        // even if the remote config.json changed since it was downloaded).
        val downloadedIds = downloaded.map { it.id }.toSet()
        val remoteNotYetDownloaded = remote.filterNot { it.id in downloadedIds }

        return bundled + downloaded + remoteNotYetDownloaded
    }

    /** Read a downloaded model's config.json straight from local storage — no network. */
    private fun loadDownloadedDescriptor(context: Context, modelId: String): ModelDescriptor {
        val configFile = ModelDownloadManager.localConfigFile(context, modelId)
        val obj = JSONObject(configFile.readText())
        val descriptor = parseDescriptor(obj, ModelDownloadStatus.DOWNLOADED)
        descriptorCache[descriptor.id] = descriptor
        return descriptor
    }

    /** Convert a GitHub-listed model into a descriptor, marking it downloaded if it's already local. */
    private fun toDescriptor(context: Context, remote: GitHubModelSource.RemoteModel): ModelDescriptor {
        val modelFileName = remote.config.optString("model_file_name", "model.onnx")
        val status = if (ModelDownloadManager.isDownloaded(context, remote.id, modelFileName)) {
            ModelDownloadStatus.DOWNLOADED
        } else {
            ModelDownloadStatus.AVAILABLE_REMOTE
        }
        val descriptor = parseDescriptor(remote.config, status)
        descriptorCache[descriptor.id] = descriptor
        return descriptor
    }

    /** Read + cache a bundled (assets/) model's config.json as a [ModelDescriptor]. */
    private fun loadBundledDescriptor(context: Context, modelId: String): ModelDescriptor {
        descriptorCache[modelId]?.let { return it }
        val configPath = "models/$modelId/config.json"
        val json = context.assets.open(configPath).bufferedReader().use { it.readText() }
        val descriptor = parseDescriptor(JSONObject(json), ModelDownloadStatus.BUNDLED)
        descriptorCache[modelId] = descriptor
        return descriptor
    }

    /**
     * Re-read a model's descriptor after it's been downloaded, so its status
     * flips from AVAILABLE_REMOTE to DOWNLOADED. Call this after a successful
     * [ModelDownloadManager.download].
     */
    fun refreshAfterDownload(context: Context, modelId: String): ModelDescriptor? {
        val configFile = ModelDownloadManager.localConfigFile(context, modelId)
        if (!configFile.exists()) return null
        val obj = JSONObject(configFile.readText())
        val descriptor = parseDescriptor(obj, ModelDownloadStatus.DOWNLOADED)
        descriptorCache[modelId] = descriptor
        return descriptor
    }

    private fun parseDescriptor(obj: JSONObject, status: ModelDownloadStatus): ModelDescriptor {
        val labels = mutableListOf<String>()
        obj.optJSONArray("labels")?.let { arr ->
            for (i in 0 until arr.length()) labels.add(arr.getString(i))
        }

        return ModelDescriptor(
            id = obj.getString("id"),
            displayNameAr = obj.getString("display_name_ar"),
            displayNameEn = obj.getString("display_name_en"),
            modality = Modality.valueOf(obj.getString("modality").uppercase()),
            task = ModelTask.valueOf(obj.getString("task").uppercase()),
            descriptionAr = obj.optString("description_ar", ""),
            inputWidth = obj.getInt("input_width"),
            inputHeight = obj.getInt("input_height"),
            fileSizeMb = obj.optInt("file_size_mb", 0),
            labels = labels,
            modelFileName = obj.optString("model_file_name", "model.onnx"),
            multiLabel = obj.optBoolean("multi_label", false),
            downloadStatus = status
        )
    }

    fun descriptorById(modelId: String): ModelDescriptor? = descriptorCache[modelId]

    /**
     * Like [descriptorById], but if the descriptor isn't cached yet (e.g.
     * this is a fresh process and [listAvailable] hasn't run this session),
     * falls back to reading the model straight from local storage — no
     * network call. This is what lets InferenceActivity open a previously
     * downloaded model directly and fully offline, even without visiting
     * the model list screen first in this session.
     */
    fun ensureDescriptor(context: Context, modelId: String): ModelDescriptor? {
        descriptorCache[modelId]?.let { return it }
        return runCatching { loadDownloadedDescriptor(context, modelId) }.getOrNull()
            ?: runCatching { loadBundledDescriptor(context, modelId) }.getOrNull()
    }

    /**
     * Create a ready-to-load [MedicalModel] instance for the given id.
     * The model's files must already be downloaded (or bundled) — check
     * [ModelDescriptor.downloadStatus] before calling this.
     * Dispatches on [ModelTask] so each task family gets the right
     * pre/post-processing implementation.
     */
    fun instantiate(context: Context, modelId: String): MedicalModel {
        val descriptor = descriptorCache[modelId]
            ?: throw IllegalStateException(
                "No descriptor cached for $modelId — call listAvailable() first."
            )
        return when (descriptor.task) {
            ModelTask.CLASSIFICATION -> OnnxClassificationModel(descriptor)

            // When you add segmentation/anomaly-detection models, implement
            // MedicalModel for them (see ai/models/) and dispatch here, e.g.:
            // ModelTask.SEGMENTATION -> OnnxSegmentationModel(descriptor)
            // ModelTask.ANOMALY_DETECTION -> OnnxAnomalyModel(descriptor)

            else -> throw UnsupportedOperationException(
                "No implementation registered yet for task ${descriptor.task}. " +
                "Add one under ai/models/ and wire it up in ModelRegistry.instantiate()."
            )
        }
    }
}
