package com.medapp.ai.remote

import android.content.Context
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Handles downloading a model's files (config.json + model.onnx) from GitHub
 * into app-private local storage, and checking what's already been
 * downloaded. Inference always reads from local storage — never directly
 * over the network — so once a model is downloaded it works fully offline.
 */
object ModelDownloadManager {

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS) // model files can be large
        .build()

    /** Local folder where downloaded models live: <filesDir>/models/<model_id>/ */
    private fun modelDir(context: Context, modelId: String): File =
        File(context.filesDir, "models/$modelId").apply { mkdirs() }

    /** True if this model's files are already downloaded and ready to use offline. */
    fun isDownloaded(context: Context, modelId: String, modelFileName: String): Boolean {
        val dir = modelDir(context, modelId)
        return File(dir, "config.json").exists() && File(dir, modelFileName).exists()
    }

    fun localConfigFile(context: Context, modelId: String): File =
        File(modelDir(context, modelId), "config.json")

    fun localModelFile(context: Context, modelId: String, modelFileName: String): File =
        File(modelDir(context, modelId), modelFileName)

    /**
     * List every model that's already downloaded to this device, by
     * scanning local storage only — no network call. This is what makes
     * previously-downloaded models still usable while fully offline.
     */
    fun listDownloadedModelIds(context: Context): List<String> {
        val modelsRoot = File(context.filesDir, "models")
        if (!modelsRoot.exists()) return emptyList()
        return modelsRoot.listFiles { file -> file.isDirectory }
            ?.filter { dir -> File(dir, "config.json").exists() }
            ?.map { dir -> dir.name }
            ?: emptyList()
    }

    /**
     * Downloads a remote model's config + weight file into local storage.
     * Call from a background thread/coroutine — this blocks on network I/O.
     *
     * @param onProgress called with (bytesDownloaded, totalBytes) while the
     *   model file streams in; totalBytes is -1 if the server didn't send a
     *   Content-Length header.
     */
    @Throws(IOException::class)
    fun download(
        context: Context,
        remoteModel: GitHubModelSource.RemoteModel,
        onProgress: (Long, Long) -> Unit = { _, _ -> }
    ) {
        val dir = modelDir(context, remoteModel.id)
        val modelFileName = remoteModel.config.optString("model_file_name", "model.onnx")

        // Download into temp files first, only replacing the real files once
        // fully written, so a failed/interrupted download can't leave a
        // half-written model file that looks "present" but is corrupt.
        val tmpConfig = File(dir, "config.json.tmp")
        val tmpModel = File(dir, "$modelFileName.tmp")

        downloadToFile(remoteModel.configUrl, tmpConfig)
        downloadToFile(remoteModel.modelFileUrl, tmpModel, onProgress)

        tmpConfig.copyTo(File(dir, "config.json"), overwrite = true)
        tmpModel.copyTo(File(dir, modelFileName), overwrite = true)
        tmpConfig.delete()
        tmpModel.delete()
    }

    /** Deletes a downloaded model's local files to free up space. */
    fun delete(context: Context, modelId: String) {
        modelDir(context, modelId).deleteRecursively()
    }

    @Throws(IOException::class)
    private fun downloadToFile(
        url: String,
        destination: File,
        onProgress: (Long, Long) -> Unit = { _, _ -> }
    ) {
        val request = Request.Builder().url(url).build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("Download failed for $url (${response.code})")
            }
            val body = response.body ?: throw IOException("Empty response body for $url")
            val total = body.contentLength()
            var downloaded = 0L

            body.byteStream().use { input ->
                destination.outputStream().use { output ->
                    val buffer = ByteArray(8 * 1024)
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        downloaded += read
                        onProgress(downloaded, total)
                    }
                }
            }
        }
    }
}
