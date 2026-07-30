"""
تحويل نموذج DenseNet121 من مكتبة TorchXRayVision إلى صيغة ONNX تعمل على
أندرويد بدون إنترنت.

TorchXRayVision (https://github.com/mlmed/torchxrayvision) مكتبة متخصصة في
أشعة الصدر، وعندها أوزان DenseNet121 جاهزة ومُدرَّبة فعليًا على datasets
حقيقية (NIH ChestX-ray8, CheXpert, RSNA Pneumonia, PadChest, MIMIC-CXR).
هذا السكريبت بيحمّل الوزن المختار تلقائيًا ويحوله لـ ONNX.

الاستخدام:
    pip install torchxrayvision torch onnx onnxruntime
    python convert_to_onnx.py --weights densenet121-res224-nih --output model.onnx

الأوزان المتاحة (كل واحد فيهم مدرب على dataset مختلف):
    densenet121-res224-all    -> كل الداتاسيتس مجمّعة
    densenet121-res224-nih    -> NIH ChestX-ray8
    densenet121-res224-rsna   -> RSNA Pneumonia Challenge
    densenet121-res224-pc     -> PadChest
    densenet121-res224-chex   -> CheXpert (Stanford)
    densenet121-res224-mimic_nb / mimic_ch -> MIMIC-CXR (MIT)

ملاحظة مهمة: مش كل وزن مدرب على كل الـ 18 تصنيف. راجع model.pathologies
بعد التحميل عشان تعرف التصنيفات الصحيحة فعليًا لنفس الوزن اللي اخترته،
وحدّث حقل "labels" في config.json بنفس القائمة بالظبط (بنفس الترتيب).
"""

import argparse

import torch


def convert(weights_name: str, output_path: str):
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=weights_name)
    model.eval()

    # اطبع التصنيفات الفعلية الصحيحة لهذا الوزن — استخدمها في config.json
    print("التصنيفات (pathologies) الخاصة بهذا الوزن بالترتيب:")
    print(model.pathologies)

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
    print("الخطوة التالية: انسخ الملف إلى")
    print("  app/src/main/assets/models/<model_id>/model.onnx")
    print("\nتذكير: عدّل config.json الخاص بالنموذج ليطابق:")
    print("  - labels: انسخ قائمة model.pathologies المطبوعة فوق بالضبط")
    print("  - input_width / input_height: 224")
    print("  - ملاحظة: هذا نموذج رمادي القناة (1 channel) وليس RGB (3 قنوات)،")
    print("    يعني كود المعالجة في التطبيق (Kotlin) لازم يستخدم")
    print("    bitmapToGrayscaleFloatBuffer بدل bitmapToNchwFloatBuffer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="densenet121-res224-nih",
        help="اسم الوزن المُدرَّب (راجع القائمة في أعلى الملف)",
    )
    parser.add_argument("--output", default="model.onnx", help="مسار ملف الإخراج")
    args = parser.parse_args()

    convert(args.weights, args.output)

