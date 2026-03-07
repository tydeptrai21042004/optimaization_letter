from __future__ import annotations

import torch.nn as nn


def _import_torchvision_models():
    try:
        from torchvision import models
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "torchvision models could not be imported. In Kaggle, install a compatible torch/torchvision pair. "
            f"Original error: {exc}"
        ) from exc
    return models


def maybe_adapt_resnet_for_small_images(model_name: str, model: nn.Module, input_size: int, pretrained: bool):
    if model_name in {"resnet18", "resnet34", "resnet50"} and input_size == 32 and not pretrained:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def replace_classifier(model_name: str, model: nn.Module, num_classes: int):
    if model_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def build_model(model_name: str, num_classes: int, pretrained: bool, input_size: int):
    models = _import_torchvision_models()
    model_name = model_name.lower()
    try:
        if model_name == "resnet18":
            w = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=w)
        elif model_name == "resnet34":
            w = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=w)
        elif model_name == "resnet50":
            w = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=w)
        elif model_name == "mobilenet_v3_small":
            w = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=w)
        elif model_name == "efficientnet_b0":
            w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=w)
        elif model_name == "vit_b_16":
            w = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.vit_b_16(weights=w)
        else:
            raise ValueError(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Build model failed: model={model_name}, pretrained={pretrained}. Error: {exc}"
        ) from exc

    model = maybe_adapt_resnet_for_small_images(model_name, model, input_size, pretrained)
    model = replace_classifier(model_name, model, num_classes)
    return model
