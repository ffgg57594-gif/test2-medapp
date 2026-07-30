"""
تحويل نموذج DenseNet121 (مُدرَّب على CheXpert - Stanford) من مكتبة
TorchXRayVision إلى صيغة ONNX تعمل على أندرويد بدون إنترنت.

هذا نموذج إضافي لأشعة الصدر (chest X-ray)، من نفس عائلة الموديل الموجود
حاليًا في التطبيق (densenet121_chest_xray المدرب على NIH)، لكنه مُدرَّب على
بيانات CheXpert من Stanford. الهدف إضافته كموديل ثانٍ للمقارنة، مش استبدال
الموديل الحالي.

الاستخدام:
    pip install torchxrayvision torch onnx onnxruntime
    python scripts/convert_chexpert_to_onnx.py --output model.onnx

ملاحظة مهمة: هذا سكريبت مخصص لوزن "densenet121-res224-chex" تحديدًا.
لو عايز تجرب أوزان تانية من نفس المكتبة، استخدم السكريبت العام
scripts/convert_to_onnx.py مع --weights.
"""

import argparse

import torch


WEIGHTS_NAME = "densenet121-res224-chex"


def convert(output_path: str):
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=WEIGHTS_NAME)
    model.eval()

    # اطبع التصنيفات الفعلية الصحيحة لهذا الوزن — استخدمها حرفيًا في config.json
    pathologies = list(model.pathologies)
    print("التصنيفات (pathologies) الخاصة بهذا الوزن بالترتيب:")
    print(pathologies)

    # تحذير: بعض أوزان torchxrayvision بتحتوي على خانات فاضية "" في
    # pathologies لو الـ dataset الأصلي مغطاش كل التصنيفات المشتركة بين
    # كل الأوزان. لازم تتأكد من القائمة دي قبل ما تحطها في config.json.
    empty_slots = [i for i, p in enumerate(pathologies) if not p]
    if empty_slots:
        print(
            f"\n⚠️  تنبيه: في {len(empty_slots)} خانة/خانات فاضية في "
            f"pathologies عند index: {empty_slots}."
        )
        print(
            "   لازم تحذف الـ index دول (أو تتعامل معاهم) في config.json، "
            "لأن الموديل هيرجع output لكل الخانات لكن الفاضية دي مالهاش "
            "معنى سريري ومينفعش تتعرض للمستخدم."
        )

    # مدخل TorchXRayVision: صورة رمادية (1 قناة) بحجم 224x224، مُطبَّعة بين -1024 و 1024
    dummy_input = torch.randn(1, 1, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # شكل ثابت أبسط وأسرع للاستدلال على الموبايل
    )

    print(f"\nتم التصدير بنجاح إلى: {output_path}")
    print("الخطوة التالية:")
    print("  1) أنشئ فولدر جديد:")
    print("     app/src/main/assets/models/densenet121_chest_xray_chexpert/")
    print(f"  2) انسخ {output_path} داخل الفولدر باسم model.onnx")
    print("  3) اكتب config.json يطابق:")
    print("     - labels: انسخ قائمة pathologies المطبوعة فوق (بعد إزالة أي خانة فاضية)")
    print("     - input_width / input_height: 224")
    print("     - multi_label: true (نفس منطق النموذج الأصلي - حالات متعددة ممكنة في نفس الوقت)")
    print("  4) ضيف السطر التالي في MODEL_IDS داخل ModelRegistry.kt:")
    print('     "densenet121_chest_xray_chexpert"')
    print(
        "\nتذكير: هذا نموذج رمادي القناة (1 channel) وليس RGB، يعني الكود "
        "بيستخدم bitmapToGrayscaleFloatBuffer تلقائيًا لأنه من نفس فئة "
        "OnnxClassificationModel — مفيش تعديل كود مطلوب."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="model.onnx", help="مسار ملف الإخراج")
    args = parser.parse_args()

    convert(args.output)
