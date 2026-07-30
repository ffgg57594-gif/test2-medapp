package com.medapp.ui.screens

import android.content.Intent
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.medapp.R
import com.medapp.ai.core.ModelDescriptor
import com.medapp.ai.registry.ModelRegistry
import com.medapp.ai.remote.GitHubModelSource
import com.medapp.ai.remote.ModelDownloadManager
import com.medapp.databinding.ActivityModelListBinding
import com.medapp.ui.components.ModelListAdapter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Shows every model currently available: bundled models plus whatever is
 * currently published on GitHub. This screen requires zero changes when new
 * models are published there — it always reflects whatever ModelRegistry
 * discovers at refresh time.
 *
 * Fetching the list involves a network call (GitHub API), so it always runs
 * on a background thread via lifecycleScope + Dispatchers.IO.
 */
class ModelListActivity : AppCompatActivity() {

    private lateinit var binding: ActivityModelListBinding
    private var adapter: ModelListAdapter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityModelListBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.recyclerModels.layoutManager = LinearLayoutManager(this)
        refreshModelList()
    }

    private fun refreshModelList() {
        setListLoading(true)
        lifecycleScope.launch {
            val models = withContext(Dispatchers.IO) {
                ModelRegistry.listAvailable(this@ModelListActivity)
            }
            setListLoading(false)

            adapter = ModelListAdapter(
                models = models,
                onModelClick = { descriptor -> openInference(descriptor) },
                onDownloadClick = { descriptor -> downloadModel(descriptor) }
            )
            binding.recyclerModels.adapter = adapter

            binding.tvEmptyState.visibility =
                if (models.isEmpty()) android.view.View.VISIBLE else android.view.View.GONE
        }
    }

    private fun openInference(descriptor: ModelDescriptor) {
        val intent = Intent(this, InferenceActivity::class.java).apply {
            putExtra(InferenceActivity.EXTRA_MODEL_ID, descriptor.id)
        }
        startActivity(intent)
    }

    private fun downloadModel(descriptor: ModelDescriptor) {
        lifecycleScope.launch {
            try {
                // Re-fetch this model's remote listing (we need its download
                // URLs, which aren't stored on ModelDescriptor itself).
                val remoteModel = withContext(Dispatchers.IO) {
                    GitHubModelSource.fetchAvailableModels()
                        .firstOrNull { it.id == descriptor.id }
                } ?: run {
                    Toast.makeText(this@ModelListActivity, R.string.download_failed, Toast.LENGTH_SHORT).show()
                    return@launch
                }

                withContext(Dispatchers.IO) {
                    ModelDownloadManager.download(
                        context = this@ModelListActivity,
                        remoteModel = remoteModel,
                        onProgress = { downloaded, total ->
                            if (total > 0) {
                                val percent = ((downloaded * 100) / total).toInt()
                                runOnUiThread { adapter?.setDownloadProgress(descriptor.id, percent) }
                            }
                        }
                    )
                }

                ModelRegistry.refreshAfterDownload(this@ModelListActivity, descriptor.id)
                adapter?.setDownloadProgress(descriptor.id, null)
                refreshModelList() // reload so the card switches to "ready to use"
            } catch (e: Exception) {
                adapter?.setDownloadProgress(descriptor.id, null)
                Toast.makeText(this@ModelListActivity, R.string.download_failed, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun setListLoading(loading: Boolean) {
        binding.progressLoadingList.visibility = if (loading) android.view.View.VISIBLE else android.view.View.GONE
        binding.tvEmptyState.visibility = android.view.View.GONE
    }
}
