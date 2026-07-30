package com.medapp.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.LayoutInflater
import android.view.View
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.medapp.R
import com.medapp.ai.core.InferenceResult
import com.medapp.ai.core.LabelScore
import com.medapp.ai.core.MedicalModel
import com.medapp.ai.registry.ModelRegistry
import com.medapp.databinding.ActivityInferenceBinding
import com.medapp.databinding.ItemResultRowBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Generic inference screen: works for ANY model task type because it only
 * talks to the [MedicalModel] interface and branches on [InferenceResult]'s
 * sealed subtype to decide how to render the outcome.
 *
 * Supports picking the source image either from the camera or from the
 * device gallery. Whichever image is loaded, it's shown in the preview
 * card and can be analyzed once a model is ready.
 */
class InferenceActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_MODEL_ID = "extra_model_id"
        private const val CAMERA_PERMISSION_REQUEST = 1001
    }

    private lateinit var binding: ActivityInferenceBinding
    private var model: MedicalModel? = null
    private var selectedBitmap: Bitmap? = null
    private var pendingCameraUri: Uri? = null

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { loadAndShowImage(it) }
    }

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        if (success) {
            pendingCameraUri?.let { loadAndShowImage(it) }
        }
    }

    private val requestCameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted: Boolean ->
        if (granted) launchCamera()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityInferenceBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val modelId = intent.getStringExtra(EXTRA_MODEL_ID)
            ?: error("InferenceActivity requires EXTRA_MODEL_ID")

        setLoadingModel(true)
        lifecycleScope.launch {
            // Falls back to reading the model straight from local storage
            // if it isn't already cached from the model-list screen this
            // session — keeps this screen fully usable offline.
            val descriptor = withContext(Dispatchers.IO) {
                ModelRegistry.ensureDescriptor(this@InferenceActivity, modelId)
            }
            if (descriptor == null) {
                setLoadingModel(false)
                binding.tvResult.text = "⚠️ تعذر تحميل بيانات النموذج، ارجع لقائمة النماذج وحاول مرة أخرى"
                return@launch
            }
            binding.tvTitle.text = descriptor.displayNameAr
            binding.tvDisclaimer.text = getString(R.string.medical_disclaimer)

            model = ModelRegistry.instantiate(this@InferenceActivity, modelId)
            model?.load(this@InferenceActivity)
            setLoadingModel(false)
        }

        binding.btnTakePhoto.setOnClickListener { onTakePhotoClicked() }
        binding.btnPickImage.setOnClickListener { pickImageLauncher.launch("image/*") }
        binding.btnAnalyze.setOnClickListener { runInference() }
    }

    // --- Camera capture -----------------------------------------------

    private fun onTakePhotoClicked() {
        val hasPermission = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (hasPermission) {
            launchCamera()
        } else {
            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun launchCamera() {
        val capturesDir = File(cacheDir, "captures").apply { mkdirs() }
        val photoFile = File(capturesDir, "capture_${System.currentTimeMillis()}.jpg")
        val uri = FileProvider.getUriForFile(
            this, "${packageName}.fileprovider", photoFile
        )
        pendingCameraUri = uri
        takePictureLauncher.launch(uri)
    }

    // --- Image loading ---------------------------------------------------

    private fun loadAndShowImage(uri: Uri) {
        val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
        selectedBitmap = bitmap
        binding.ivPreview.setImageBitmap(bitmap)
        binding.tvNoImageHint.visibility = View.GONE
        binding.btnAnalyze.isEnabled = model != null
        binding.cardResults.visibility = View.GONE
        binding.tvResult.text = ""
    }

    // --- Inference ---------------------------------------------------

    private fun runInference() {
        val bitmap = selectedBitmap ?: return
        val activeModel = model ?: return

        setAnalyzing(true)
        lifecycleScope.launch {
            val result = activeModel.runInference(bitmap)
            setAnalyzing(false)
            renderResult(result)
        }
    }

    private fun renderResult(result: InferenceResult) {
        when (result) {
            is InferenceResult.Classification -> renderClassification(result)

            is InferenceResult.Segmentation -> {
                binding.ivPreview.setImageBitmap(result.maskBitmap)
                binding.cardResults.visibility = View.VISIBLE
                binding.tvTopResult.text =
                    "نسبة المنطقة المتأثرة: ${"%.1f".format(result.affectedAreaPercent)}%"
                binding.tvInferenceTime.text = "زمن التحليل: ${result.inferenceTimeMs} ملي ثانية"
                binding.resultsListContainer.removeAllViews()
                binding.tvResult.text = ""
            }

            is InferenceResult.Error -> {
                binding.cardResults.visibility = View.GONE
                binding.tvResult.text = "⚠️ ${result.message}"
            }
        }
    }

    private fun renderClassification(result: InferenceResult.Classification) {
        binding.tvResult.text = ""
        val sorted = result.predictions.sortedByDescending { it.confidence }
        val top = sorted.firstOrNull()

        binding.tvTopResult.text = top?.let {
            "الأرجح: ${it.label} — ${(it.confidence * 100).toInt()}%"
        } ?: "لا توجد نتيجة"
        binding.tvInferenceTime.text = "زمن التحليل: ${result.inferenceTimeMs} ملي ثانية"

        binding.resultsListContainer.removeAllViews()
        sorted.forEach { prediction ->
            binding.resultsListContainer.addView(buildResultRow(prediction))
        }
        binding.cardResults.visibility = View.VISIBLE
    }

    private fun buildResultRow(prediction: LabelScore): View {
        val rowBinding = ItemResultRowBinding.inflate(
            LayoutInflater.from(this), binding.resultsListContainer, false
        )
        val percent = (prediction.confidence * 100).toInt()

        rowBinding.tvLabel.text = prediction.label
        rowBinding.tvPercent.text = "$percent%"
        rowBinding.progressConfidence.progress = percent

        val color = when {
            percent >= 60 -> ContextCompat.getColor(this, R.color.confidence_high)
            percent >= 30 -> ContextCompat.getColor(this, R.color.confidence_medium)
            else -> ContextCompat.getColor(this, R.color.confidence_low)
        }
        rowBinding.tvPercent.setTextColor(color)
        rowBinding.progressConfidence.progressTintList = android.content.res.ColorStateList.valueOf(color)

        return rowBinding.root
    }

    // --- Loading state helpers ---------------------------------------------------

    /** Shown briefly while the ONNX session is created from the local model file. */
    private fun setLoadingModel(loading: Boolean) {
        binding.loadingContainer.visibility = if (loading) View.VISIBLE else View.GONE
        binding.progressBar.isIndeterminate = true
        binding.tvLoadingMessage.text = if (loading) "جاري تجهيز النموذج…" else ""
        binding.btnAnalyze.isEnabled = !loading && selectedBitmap != null
    }

    private fun setAnalyzing(analyzing: Boolean) {
        binding.loadingContainer.visibility = if (analyzing) View.VISIBLE else View.GONE
        binding.progressBar.isIndeterminate = true
        binding.tvLoadingMessage.text = if (analyzing) getString(R.string.analyzing_in_progress) else ""
        binding.btnAnalyze.isEnabled = !analyzing
        binding.btnTakePhoto.isEnabled = !analyzing
        binding.btnPickImage.isEnabled = !analyzing
    }

    override fun onDestroy() {
        super.onDestroy()
        model?.unload()
    }
}
