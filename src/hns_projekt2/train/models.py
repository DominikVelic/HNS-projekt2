from torch import nn
import timm
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel

MODELS_KEYS = {
    # CNN
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k", # A ConvNeXt image classification model. Pretrained on ImageNet-22k and fine-tuned on ImageNet-1k by paper authors.
    "yolo26n": "yolo26n-cls", # Yolo 26 nano classification

    # ViT
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k", # A Vision Transformer (ViT) image classification model. Trained on ImageNet-21k and fine-tuned on ImageNet-1k (with additional augmentation and regularization) in JAX by paper authors, ported to PyTorch by Ross Wightman.
    # "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k", # A Swin Transformer image classification model. Pretrained on ImageNet-1k by paper authors.

    # Hybrid (convolution + attention)
    "coatnet_0": "coatnet_0_rw_224.sw_in1k", # A timm specific CoAtNet image classification model. Trained in timm on ImageNet-1k by Ross Wightman.
    # "mobilevit_s": "mobilevit_s.cvnets_in1k", # A MobileViT image classification model. Trained on ImageNet-1k by paper authors.
}

MODEL_NAMES = {
    "yolo26n": "Yolo26 nano",
    "convnext_tiny": "ConvNeXt-Tiny",

    "vit_small": "ViT small",
    "swin_tiny": "Swin transformer image classification model",

    "coatnet_0": "CoAtNet-0",
    "mobilevit_s": "MobileViT Small",
}

# Same preprocessing Ultralytics uses for classification: 224px, pixels / 255, no mean/std
YOLO_DATA_CFG = dict(input_size=(3, 224, 224), interpolation="bilinear",
                     mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), crop_pct=1.0)


class YOLOClassifier(nn.Module):
    """Plain nn.Module around a YOLO classification net, so it trains in our own loop."""

    def __init__(self, weights: str = "yolo26n-cls.pt", num_classes: int = 2):
        super().__init__()
        self.net = YOLO(weights).model.float()      # checkpoints are stored in fp16
        ClassificationModel.reshape_outputs(self.net, num_classes)  # 1000 -> num_classes head
        for p in self.net.parameters():             # some loaded layers can be frozen
            p.requires_grad = True

    def forward(self, x):
        out = self.net(x)
        # Classify head returns logits in train mode, (probs, logits) in eval mode
        return out[1] if isinstance(out, tuple) else out


def build_model(key, num_classes: int = 2, pretrained: bool = True):
    """Returns (model, data_cfg); data_cfg describes the input preprocessing the model expects."""
    if key not in MODELS_KEYS:
        raise ValueError(f"Neznámy model: {key}")
    try:
        if key == "yolo26n":
            return YOLOClassifier(MODELS_KEYS[key] + ".pt", num_classes), YOLO_DATA_CFG
        model = timm.create_model(MODELS_KEYS[key], pretrained=pretrained, num_classes=num_classes)
    except Exception as exc:
        raise RuntimeError(
            f"Nepodarilo sa vytvoriť {MODEL_NAMES.get(key, key)}. "
            "Skontroluj verziu timm a internetové pripojenie pre predtrénované váhy. "
            f"Pôvodná chyba: {exc}"
        ) from exc
    return model, timm.data.resolve_model_data_config(model)


if __name__ == "__main__":
    model, _ = build_model("convnext_tiny")
    for module in reversed(list(model.modules())):
        print(module)
