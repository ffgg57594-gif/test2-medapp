package com.medapp.ui.components

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.medapp.ai.core.ModelDescriptor
import com.medapp.ai.core.ModelDownloadStatus
import com.medapp.ai.core.ModelTask
import com.medapp.databinding.ItemModelBinding

/**
 * Renders the list of available models generically from their descriptors.
 * Adding a new model to the registry (or publishing one on GitHub — see
 * ModelRegistry) automatically makes it show up here, no changes needed.
 *
 * Each model can be in one of three states (see [ModelDownloadStatus]):
 *  - DOWNLOADED / BUNDLED: tapping the card opens it for inference
 *  - AVAILABLE_REMOTE: shows a "Download" button instead
 *  - mid-download: shows a progress bar (tracked per model id in [downloadProgress])
 */
class ModelListAdapter(
    private val models: List<ModelDescriptor>,
    private val onModelClick: (ModelDescriptor) -> Unit,
    private val onDownloadClick: (ModelDescriptor) -> Unit
) : RecyclerView.Adapter<ModelListAdapter.ViewHolder>() {

    /** modelId -> progress percent (0-100), or null if not currently downloading. */
    private val downloadProgress = mutableMapOf<String, Int>()

    inner class ViewHolder(val binding: ItemModelBinding) : RecyclerView.ViewHolder(binding.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemModelBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val model = models[position]
        val b = holder.binding

        b.tvModelName.text = model.displayNameAr
        b.tvModelDescription.text = model.descriptionAr
        b.tvModelBadge.text = taskLabel(model.task)
        b.tvModelSize.text = "${model.fileSizeMb} MB"

        val progress = downloadProgress[model.id]

        when {
            progress != null -> {
                // Mid-download: show progress, hide download button, disable tap.
                b.progressDownload.visibility = View.VISIBLE
                b.progressDownload.progress = progress
                b.btnDownload.visibility = View.GONE
                b.root.setOnClickListener(null)
            }
            model.downloadStatus == ModelDownloadStatus.AVAILABLE_REMOTE -> {
                // Needs downloading first.
                b.progressDownload.visibility = View.GONE
                b.btnDownload.visibility = View.VISIBLE
                b.btnDownload.isEnabled = true
                b.btnDownload.setOnClickListener { onDownloadClick(model) }
                b.root.setOnClickListener(null)
            }
            else -> {
                // BUNDLED or DOWNLOADED: ready to use.
                b.progressDownload.visibility = View.GONE
                b.btnDownload.visibility = View.GONE
                b.root.setOnClickListener { onModelClick(model) }
            }
        }
    }

    override fun getItemCount(): Int = models.size

    /** Update the progress bar for a model mid-download. Pass null when download finishes/fails. */
    fun setDownloadProgress(modelId: String, percent: Int?) {
        if (percent == null) downloadProgress.remove(modelId) else downloadProgress[modelId] = percent
        val index = models.indexOfFirst { it.id == modelId }
        if (index != -1) notifyItemChanged(index)
    }

    private fun taskLabel(task: ModelTask): String = when (task) {
        ModelTask.CLASSIFICATION -> "تصنيف"
        ModelTask.SEGMENTATION -> "تجزئة"
        ModelTask.ANOMALY_DETECTION -> "كشف شذوذ"
    }
}
