# إضافة نموذج جديد للتطبيق

الهيكل مصمم بحيث إضافة نموذج تصنيف (classification) جديد لا تحتاج تعديل أي
كود UI. اتبع الخطوات دي:

## 1. حوّل النموذج إلى ONNX

استخدم `scripts/convert_to_onnx.py` كمثال. أي نموذج PyTorch/MONAI لازم
يتحول لـ ONNX عشان يشتغل على أندرويد offline عبر ONNX Runtime.

```bash
python scripts/convert_to_onnx.py --checkpoint my_model.pt --output model.onnx
```

## 2. أنشئ مجلد النموذج

```
app/src/main/assets/models/<model_id>/
  ├── model.onnx
  └── config.json
```

`<model_id>` لازم يكون اسم فريد بدون مسافات، مثلاً `densenet_bone_xray`.

## 3. اكتب config.json

انسخ `config.json` بتاع `densenet121_chest_xray` كقالب وعدّل القيم:

```json
{
  "id": "densenet_bone_xray",
  "display_name_ar": "تصنيف أشعة العظام",
  "display_name_en": "Bone X-ray Classifier",
  "modality": "xray",
  "task": "classification",
  "description_ar": "وصف مختصر...",
  "input_width": 224,
  "input_height": 224,
  "file_size_mb": 30,
  "labels": ["طبيعي", "كسر", "أخرى"]
}
```

`modality` الخيارات المتاحة: `xray`, `ct`, `mri`, `pathology`, `other`
`task` الخيارات المتاحة حاليًا: `classification` (تصنيف).
segmentation و anomaly_detection محتاجين implementation إضافي — راجع القسم
التالي.

حقل اختياري مهم: `multi_label` (true/false, افتراضيًا false):
- استخدم `true` لو النموذج بيكشف عدة حالات مستقلة عن بعضها في نفس الوقت
  (زي TorchXRayVision — ممكن المريض يكون عنده أكتر من حالة في نفس الوقت).
  التطبيق هيستخدم sigmoid لكل تصنيف على حدة.
- استخدم `false` (أو سيبه بره) لو النموذج بيختار تصنيف واحد بس من بين عدة
  خيارات متنافية. التطبيق هيستخدم softmax.

## 4. سجّل النموذج

في `ModelRegistry.kt`، ضيف سطر واحد بس:

```kotlin
private val MODEL_IDS = listOf(
    "densenet121_chest_xray",
    "densenet_bone_xray"   // <-- السطر الجديد
)
```

كده خلاص. النموذج هيظهر تلقائيًا في شاشة اختيار النماذج، وهيشتغل بنفس شاشة
التحليل الموجودة، من غير أي تعديل تاني.

---

## إضافة نوع مهمة جديد (Segmentation / Anomaly Detection)

لو النموذج مش تصنيف بسيط (يعني بيرجع mask أو منطقة محددة بدل نسبة مئوية)،
محتاج خطوة إضافية:

1. أنشئ class جديد في `ai/models/segmentation/` (أو مجلد مشابه) بيعمل
   implement لـ `MedicalModel` — استخدم `OnnxClassificationModel.kt`
   كمرجع للبنية العامة.
2. في `postprocess`/`runInference`، حوّل مخرجات النموذج لـ
   `InferenceResult.Segmentation` (تقدر تضيف subtype جديد في
   `InferenceResult` لو محتاج شكل بيانات مختلف).
3. في `ModelRegistry.instantiate()`, ضيف السطر:
   ```kotlin
   ModelTask.SEGMENTATION -> OnnxSegmentationModel(descriptor)
   ```
4. شاشة `InferenceActivity` بالفعل فيها `when` block جاهز لعرض نتائج
   segmentation (`is InferenceResult.Segmentation`) — مش هتحتاج تلمسها إلا
   لو عايز تخصص العرض أكتر.

## ملاحظات مهمة قبل إضافة نماذج ثلاثية الأبعاد (CT/MRI/BraTS)

نماذج زي BraTS بتاخد volume كامل (سلسلة صور DICOM/NIfTI) مش صورة واحدة.
ده معناه:
- `MedicalModel.runInference` هيحتاج توقيع مختلف (يقبل قائمة صور أو ملف
  NIfTI بدل `Bitmap` واحد) — هيحتاج تعديل بسيط على الـ interface نفسه أو
  إضافة interface موازي `VolumetricMedicalModel`.
- حجم النموذج والذاكرة المطلوبة أكبر بكتير — اختبر على جهاز حقيقي، مش
  emulator بس.
- شاشة اختيار الصور (`InferenceActivity`) محتاجة UI مختلف لاختيار سلسلة
  صور بدل صورة واحدة.

هذا التوسع ممكن لكنه أكبر من مجرد "إضافة config.json" — خطط له كخطوة
منفصلة لما توصله.
