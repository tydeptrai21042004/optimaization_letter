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


# ResNet-style backbones where the CIFAR-style small-image stem tweak is often useful
RESNET_STYLE_MODELS = {
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext50_32x4d",
    "wide_resnet50_2",
}


def _get_weights(models, enum_name: str, pretrained: bool):
    if not pretrained:
        return None
    return getattr(models, enum_name).DEFAULT


def maybe_adapt_resnet_for_small_images(model_name: str, model: nn.Module, input_size: int, pretrained: bool):
    if model_name in RESNET_STYLE_MODELS and input_size == 32 and not pretrained:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def _replace_last_linear(module: nn.Module, num_classes: int) -> bool:
    """
    Recursively replace the last nn.Linear found in the module tree.
    This makes the code robust across many torchvision classifier layouts:
    - ResNet / RegNet / ShuffleNet: model.fc
    - DenseNet: model.classifier
    - MobileNet / EfficientNet / ConvNeXt / VGG / AlexNet / MNASNet: classifier[-1]
    - ViT / Swin: heads.head / head
    """
    children = list(module.named_children())

    for name, child in reversed(children):
        if isinstance(child, nn.Linear):
            new_layer = nn.Linear(child.in_features, num_classes)
            module._modules[name] = new_layer
            return True

        if _replace_last_linear(child, num_classes):
            return True

    return False


def replace_classifier(model_name: str, model: nn.Module, num_classes: int):
    ok = _replace_last_linear(model, num_classes)
    if not ok:
        raise ValueError(f"Could not find final Linear classifier to replace for model: {model_name}")
    return model


def build_model(model_name: str, num_classes: int, pretrained: bool, input_size: int):
    models = _import_torchvision_models()
    model_name = model_name.lower()

    # name -> (builder function name, weights enum name)
    model_specs = {
        # ResNet family
        "resnet18": ("resnet18", "ResNet18_Weights"),
        "resnet34": ("resnet34", "ResNet34_Weights"),
        "resnet50": ("resnet50", "ResNet50_Weights"),
        "resnext50_32x4d": ("resnext50_32x4d", "ResNeXt50_32X4D_Weights"),
        "wide_resnet50_2": ("wide_resnet50_2", "Wide_ResNet50_2_Weights"),

        # DenseNet
        "densenet121": ("densenet121", "DenseNet121_Weights"),

        # MobileNet
        "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights"),
        "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
        "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),

        # EfficientNet
        "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
        "efficientnet_b1": ("efficientnet_b1", "EfficientNet_B1_Weights"),
        "efficientnet_v2_s": ("efficientnet_v2_s", "EfficientNet_V2_S_Weights"),

        # ConvNeXt
        "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
        "convnext_small": ("convnext_small", "ConvNeXt_Small_Weights"),

        # RegNet
        "regnet_y_400mf": ("regnet_y_400mf", "RegNet_Y_400MF_Weights"),
        "regnet_x_400mf": ("regnet_x_400mf", "RegNet_X_400MF_Weights"),

        # ShuffleNet
        "shufflenet_v2_x0_5": ("shufflenet_v2_x0_5", "ShuffleNet_V2_X0_5_Weights"),
        "shufflenet_v2_x1_0": ("shufflenet_v2_x1_0", "ShuffleNet_V2_X1_0_Weights"),

        # MNASNet
        "mnasnet0_5": ("mnasnet0_5", "MNASNet0_5_Weights"),
        "mnasnet1_0": ("mnasnet1_0", "MNASNet1_0_Weights"),

        # Transformers
        "vit_b_16": ("vit_b_16", "ViT_B_16_Weights"),
        "swin_t": ("swin_t", "Swin_T_Weights"),

        # Classic CNNs
        "alexnet": ("alexnet", "AlexNet_Weights"),
        "vgg11": ("vgg11", "VGG11_Weights"),
        "vgg16": ("vgg16", "VGG16_Weights"),
    }

    try:
        if model_name not in model_specs:
            raise ValueError(model_name)

        builder_name, weights_enum = model_specs[model_name]
        builder = getattr(models, builder_name)
        weights = _get_weights(models, weights_enum, pretrained)

        # GoogLeNet / Inception are intentionally omitted here because they often
        # need special output handling (aux logits) in the training loop.
        model = builder(weights=weights)

    except Exception as exc:
        raise RuntimeError(
            f"Build model failed: model={model_name}, pretrained={pretrained}. Error: {exc}"
        ) from exc

    model = maybe_adapt_resnet_for_small_images(model_name, model, input_size, pretrained)
    model = replace_classifier(model_name, model, num_classes)
    return model