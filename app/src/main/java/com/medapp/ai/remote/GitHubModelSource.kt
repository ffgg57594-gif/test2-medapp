package com.medapp.ai.remote

import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Talks to the GitHub REST API to discover which models are currently
 * published under the `models/` folder of the project's repo, without
 * downloading the (potentially large) model weight files themselves.
 *
 * How this stays in sync with new models: you don't touch this file when
 * you add a model. Just push a new folder under `models/<model_id>/` in the
 * repo (with a config.json + model file inside), and it will automatically
 * show up next time the app refreshes its list.
 */
object GitHubModelSource {

    // TODO: adjust if you rename/move the repo or the models folder.
    private const val OWNER = "ffgg57594-gif"
    private const val REPO = "test2-medapp"
    private const val BRANCH = "master"
    private const val MODELS_PATH = "models"

    private const val API_CONTENTS_URL =
        "https://api.github.com/repos/$OWNER/$REPO/contents/$MODELS_PATH?ref=$BRANCH"

    private fun rawUrl(modelId: String, fileName: String): String =
        "https://raw.githubusercontent.com/$OWNER/$REPO/$BRANCH/$MODELS_PATH/$modelId/$fileName"

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .build()

    /** A model folder found in the repo, with enough info to show + download it. */
    data class RemoteModel(
        val id: String,
        val configUrl: String,
        val modelFileUrl: String,
        val config: JSONObject
    )

    /**
     * Fetch the list of model folders in the repo, and each one's config.json.
     * Runs network calls — must be called from a background thread/coroutine.
     *
     * @throws IOException on network failure, or JSONException on malformed responses.
     */
    @Throws(IOException::class)
    fun fetchAvailableModels(): List<RemoteModel> {
        val listRequest = Request.Builder().url(API_CONTENTS_URL).build()

        val entries: JSONArray = client.newCall(listRequest).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("GitHub API returned ${response.code} for model list")
            }
            val body = response.body?.string() ?: "[]"
            JSONArray(body)
        }

        val results = mutableListOf<RemoteModel>()
        for (i in 0 until entries.length()) {
            val entry = entries.getJSONObject(i)
            if (entry.optString("type") != "dir") continue // skip stray files, only folders are models

            val modelId = entry.getString("name")
            val configUrl = rawUrl(modelId, "config.json")

            val config = runCatching { fetchConfig(configUrl) }.getOrNull() ?: continue
            val modelFileName = config.optString("model_file_name", "model.onnx")

            results.add(
                RemoteModel(
                    id = modelId,
                    configUrl = configUrl,
                    modelFileUrl = rawUrl(modelId, modelFileName),
                    config = config
                )
            )
        }
        return results
    }

    @Throws(IOException::class)
    private fun fetchConfig(url: String): JSONObject {
        val request = Request.Builder().url(url).build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("Failed to fetch config at $url (${response.code})")
            }
            val body = response.body?.string() ?: throw IOException("Empty config body at $url")
            return JSONObject(body)
        }
    }
}
