from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lightweight built-in models
# ---------------------------------------------------------------------------
# These models make the benchmark usable even when torchvision is unavailable
# or when a quick CPU smoke test is needed.  They share the same build_model()
# interface as the torchvision backbones below.


class TinyCNN(nn.Module):
    """Very small CNN for fast classification/regression tests."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyMLP(nn.Module):
    """MLP image baseline.  Uses adaptive pooling so it accepts any image size."""

    def __init__(self, out_dim: int, pooled_size: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * pooled_size * pooled_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.pool(x))


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallCNN(nn.Module):
    """Stronger CNN than tiny_cnn, still fast enough for CIFAR-scale tests."""

    def __init__(self, out_dim: int, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, width),
            ConvBNAct(width, width),
            nn.MaxPool2d(2),
            ConvBNAct(width, width * 2),
            ConvBNAct(width * 2, width * 2),
            nn.MaxPool2d(2),
            ConvBNAct(width * 2, width * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.features(x), 1))


class DepthwiseSeparableCNN(nn.Module):
    """MobileNet-style lightweight backbone."""

    def __init__(self, out_dim: int, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, width, stride=1),
            ConvBNAct(width, width, groups=width),
            ConvBNAct(width, width * 2, kernel_size=1),
            nn.MaxPool2d(2),
            ConvBNAct(width * 2, width * 2, groups=width * 2),
            ConvBNAct(width * 2, width * 4, kernel_size=1),
            nn.MaxPool2d(2),
            ConvBNAct(width * 4, width * 4, groups=width * 4),
            ConvBNAct(width * 4, width * 4, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.features(x), 1))


class BasicResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.proj = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv2(self.conv1(x)) + self.proj(x), inplace=True)


class SmallResNet(nn.Module):
    """ResNet-like built-in model for scratch CIFAR/synthetic runs."""

    def __init__(self, out_dim: int, width: int = 32, blocks_per_stage: int = 2):
        super().__init__()
        self.stem = ConvBNAct(3, width)
        layers: List[nn.Module] = []
        ch = width
        for stage, out_ch in enumerate([width, width * 2, width * 4]):
            stride = 1 if stage == 0 else 2
            layers.append(BasicResidualBlock(ch, out_ch, stride=stride))
            ch = out_ch
            for _ in range(blocks_per_stage - 1):
                layers.append(BasicResidualBlock(ch, ch, stride=1))
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(ch, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.head(torch.flatten(self.features(x), 1))


class MiniViT(nn.Module):
    """Small ViT-style model with flexible image size via convolutional patching."""

    def __init__(self, out_dim: int, image_size: int = 32, patch_size: int = 4, embed_dim: int = 64, depth: int = 2, num_heads: int = 4):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"mini_vit requires image_size divisible by patch_size. Got image_size={image_size}, patch_size={patch_size}")
        self.patch = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1)
        if z.size(1) != self.pos_embed.size(1):
            raise ValueError(
                f"mini_vit was built for {self.pos_embed.size(1) - 1} patches, "
                f"but input produced {z.size(1) - 1}. Use recommended_input_size() or rebuild with this image size."
            )
        z = self.encoder(z + self.pos_embed)
        return self.head(self.norm(z[:, 0]))


class TinyUNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.enc1(x)
        z = self.enc2(skip)
        z = F.interpolate(z, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec1(z))


class UNetSmall(nn.Module):
    """Small U-Net with skip connections for segmentation experiments."""

    def __init__(self, num_classes: int, width: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBNAct(3, width), ConvBNAct(width, width))
        self.enc2 = nn.Sequential(ConvBNAct(width, width * 2, stride=2), ConvBNAct(width * 2, width * 2))
        self.bridge = nn.Sequential(ConvBNAct(width * 2, width * 4, stride=2), ConvBNAct(width * 4, width * 4))
        self.dec2 = nn.Sequential(ConvBNAct(width * 4 + width * 2, width * 2), ConvBNAct(width * 2, width * 2))
        self.dec1 = nn.Sequential(ConvBNAct(width * 2 + width, width), ConvBNAct(width, width))
        self.head = nn.Conv2d(width, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bridge(e2)
        z = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec2(torch.cat([z, e2], dim=1))
        z = F.interpolate(z, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec1(torch.cat([z, e1], dim=1))
        return self.head(z)


class FCNLite(nn.Module):
    """Fully convolutional segmentation baseline."""

    def __init__(self, num_classes: int, width: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNAct(3, width),
            ConvBNAct(width, width * 2, stride=2),
            ConvBNAct(width * 2, width * 4, stride=2),
            ConvBNAct(width * 4, width * 4),
        )
        self.classifier = nn.Conv2d(width * 4, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.classifier(z)
        return F.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False)


class AtrousBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepLabLite(nn.Module):
    """Tiny DeepLab-style model with atrous context aggregation."""

    def __init__(self, num_classes: int, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(3, width), ConvBNAct(width, width * 2, stride=2))
        self.context = nn.ModuleList([AtrousBlock(width * 2, d) for d in (1, 2, 4)])
        self.fuse = nn.Sequential(ConvBNAct(width * 2 * 3, width * 2, kernel_size=1), nn.Conv2d(width * 2, num_classes, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.fuse(torch.cat([block(z) for block in self.context], dim=1))
        return F.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False)


BUILTIN_CLASSIFICATION_MODELS = {
    "tiny_mlp",
    "tiny_cnn",
    "small_cnn",
    "depthwise_cnn",
    "small_resnet",
    "wide_small_resnet",
    "mini_vit",
}

BUILTIN_SEGMENTATION_MODELS = {
    "tiny_unet",
    "unet_small",
    "fcn_lite",
    "deeplab_lite",
}

BUILTIN_MODELS = BUILTIN_CLASSIFICATION_MODELS | BUILTIN_SEGMENTATION_MODELS


# ---------------------------------------------------------------------------
# Torchvision registries
# ---------------------------------------------------------------------------


def _import_torchvision_models():
    try:
        from torchvision import models
    except Exception as exc:
        raise RuntimeError(f"torchvision models could not be imported. Original error: {exc}") from exc
    return models


# For CIFAR-like scratch training, these can be safely adapted to 32x32 inputs.
RESNET_STYLE_MODELS = {
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
}

TORCHVISION_CLASSIFICATION_SPECS: Dict[str, Tuple[str, str]] = {
    # ResNet family
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet34": ("resnet34", "ResNet34_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "resnet101": ("resnet101", "ResNet101_Weights"),
    "resnet152": ("resnet152", "ResNet152_Weights"),
    "resnext50_32x4d": ("resnext50_32x4d", "ResNeXt50_32X4D_Weights"),
    "resnext101_32x8d": ("resnext101_32x8d", "ResNeXt101_32X8D_Weights"),
    "wide_resnet50_2": ("wide_resnet50_2", "Wide_ResNet50_2_Weights"),
    "wide_resnet101_2": ("wide_resnet101_2", "Wide_ResNet101_2_Weights"),
    # DenseNet family
    "densenet121": ("densenet121", "DenseNet121_Weights"),
    "densenet161": ("densenet161", "DenseNet161_Weights"),
    "densenet169": ("densenet169", "DenseNet169_Weights"),
    "densenet201": ("densenet201", "DenseNet201_Weights"),
    # EfficientNet family
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
    "efficientnet_b1": ("efficientnet_b1", "EfficientNet_B1_Weights"),
    "efficientnet_b2": ("efficientnet_b2", "EfficientNet_B2_Weights"),
    "efficientnet_b3": ("efficientnet_b3", "EfficientNet_B3_Weights"),
    "efficientnet_b4": ("efficientnet_b4", "EfficientNet_B4_Weights"),
    "efficientnet_b5": ("efficientnet_b5", "EfficientNet_B5_Weights"),
    "efficientnet_b6": ("efficientnet_b6", "EfficientNet_B6_Weights"),
    "efficientnet_b7": ("efficientnet_b7", "EfficientNet_B7_Weights"),
    "efficientnet_v2_s": ("efficientnet_v2_s", "EfficientNet_V2_S_Weights"),
    "efficientnet_v2_m": ("efficientnet_v2_m", "EfficientNet_V2_M_Weights"),
    "efficientnet_v2_l": ("efficientnet_v2_l", "EfficientNet_V2_L_Weights"),
    # Mobile / edge models
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights"),
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
    "mnasnet0_5": ("mnasnet0_5", "MNASNet0_5_Weights"),
    "mnasnet0_75": ("mnasnet0_75", "MNASNet0_75_Weights"),
    "mnasnet1_0": ("mnasnet1_0", "MNASNet1_0_Weights"),
    "mnasnet1_3": ("mnasnet1_3", "MNASNet1_3_Weights"),
    "shufflenet_v2_x0_5": ("shufflenet_v2_x0_5", "ShuffleNet_V2_X0_5_Weights"),
    "shufflenet_v2_x1_0": ("shufflenet_v2_x1_0", "ShuffleNet_V2_X1_0_Weights"),
    "shufflenet_v2_x1_5": ("shufflenet_v2_x1_5", "ShuffleNet_V2_X1_5_Weights"),
    "shufflenet_v2_x2_0": ("shufflenet_v2_x2_0", "ShuffleNet_V2_X2_0_Weights"),
    "squeezenet1_0": ("squeezenet1_0", "SqueezeNet1_0_Weights"),
    "squeezenet1_1": ("squeezenet1_1", "SqueezeNet1_1_Weights"),
    # Modern ConvNet family
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
    "convnext_small": ("convnext_small", "ConvNeXt_Small_Weights"),
    "convnext_base": ("convnext_base", "ConvNeXt_Base_Weights"),
    "convnext_large": ("convnext_large", "ConvNeXt_Large_Weights"),
    "regnet_x_400mf": ("regnet_x_400mf", "RegNet_X_400MF_Weights"),
    "regnet_x_800mf": ("regnet_x_800mf", "RegNet_X_800MF_Weights"),
    "regnet_x_1_6gf": ("regnet_x_1_6gf", "RegNet_X_1_6GF_Weights"),
    "regnet_x_3_2gf": ("regnet_x_3_2gf", "RegNet_X_3_2GF_Weights"),
    "regnet_y_400mf": ("regnet_y_400mf", "RegNet_Y_400MF_Weights"),
    "regnet_y_800mf": ("regnet_y_800mf", "RegNet_Y_800MF_Weights"),
    "regnet_y_1_6gf": ("regnet_y_1_6gf", "RegNet_Y_1_6GF_Weights"),
    "regnet_y_3_2gf": ("regnet_y_3_2gf", "RegNet_Y_3_2GF_Weights"),
    "maxvit_t": ("maxvit_t", "MaxVit_T_Weights"),
    # Transformer family
    "vit_b_16": ("vit_b_16", "ViT_B_16_Weights"),
    "vit_b_32": ("vit_b_32", "ViT_B_32_Weights"),
    "vit_l_16": ("vit_l_16", "ViT_L_16_Weights"),
    "vit_l_32": ("vit_l_32", "ViT_L_32_Weights"),
    "swin_t": ("swin_t", "Swin_T_Weights"),
    "swin_s": ("swin_s", "Swin_S_Weights"),
    "swin_b": ("swin_b", "Swin_B_Weights"),
    "swin_v2_t": ("swin_v2_t", "Swin_V2_T_Weights"),
    "swin_v2_s": ("swin_v2_s", "Swin_V2_S_Weights"),
    "swin_v2_b": ("swin_v2_b", "Swin_V2_B_Weights"),
    # Legacy baselines
    "alexnet": ("alexnet", "AlexNet_Weights"),
    "vgg11": ("vgg11", "VGG11_Weights"),
    "vgg11_bn": ("vgg11_bn", "VGG11_BN_Weights"),
    "vgg13": ("vgg13", "VGG13_Weights"),
    "vgg13_bn": ("vgg13_bn", "VGG13_BN_Weights"),
    "vgg16": ("vgg16", "VGG16_Weights"),
    "vgg16_bn": ("vgg16_bn", "VGG16_BN_Weights"),
    "vgg19": ("vgg19", "VGG19_Weights"),
    "vgg19_bn": ("vgg19_bn", "VGG19_BN_Weights"),
    "googlenet": ("googlenet", "GoogLeNet_Weights"),
    "inception_v3": ("inception_v3", "Inception_V3_Weights"),
}

TORCHVISION_SEGMENTATION_SPECS: Dict[str, Tuple[str, str]] = {
    "fcn_resnet50": ("fcn_resnet50", "FCN_ResNet50_Weights"),
    "fcn_resnet101": ("fcn_resnet101", "FCN_ResNet101_Weights"),
    "deeplabv3_resnet50": ("deeplabv3_resnet50", "DeepLabV3_ResNet50_Weights"),
    "deeplabv3_resnet101": ("deeplabv3_resnet101", "DeepLabV3_ResNet101_Weights"),
    "deeplabv3_mobilenet_v3_large": ("deeplabv3_mobilenet_v3_large", "DeepLabV3_MobileNet_V3_Large_Weights"),
    "lraspp_mobilenet_v3_large": ("lraspp_mobilenet_v3_large", "LRASPP_MobileNet_V3_Large_Weights"),
}

# Models with fixed classifier geometry or very aggressive early downsampling are
# safer at ImageNet-style sizes, especially with torchvision implementations.
IMAGE_NET_SIZE_MODELS = {
    "alexnet",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "googlenet",
    "maxvit_t",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "swin_t",
    "swin_s",
    "swin_b",
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
}

INCEPTION_SIZE_MODELS = {"inception_v3"}


def list_supported_models(task_type: str | None = None) -> List[str]:
    """Return supported model names, useful for docs/tests/CLI help."""

    if task_type is None:
        names = BUILTIN_MODELS | set(TORCHVISION_CLASSIFICATION_SPECS) | set(TORCHVISION_SEGMENTATION_SPECS)
    else:
        task_type = task_type.lower()
        if task_type in {"classification", "regression"}:
            names = BUILTIN_CLASSIFICATION_MODELS | set(TORCHVISION_CLASSIFICATION_SPECS)
        elif task_type == "segmentation":
            names = BUILTIN_SEGMENTATION_MODELS | set(TORCHVISION_SEGMENTATION_SPECS)
        else:
            raise ValueError("task_type must be classification, regression, segmentation, or None")
    return sorted(names)


# ---------------------------------------------------------------------------
# Head replacement and builder helpers
# ---------------------------------------------------------------------------


def _get_weights(container, enum_name: str, pretrained: bool):
    if not pretrained:
        return None
    if not hasattr(container, enum_name):
        raise ValueError(f"This torchvision version does not provide weights enum {enum_name}")
    return getattr(container, enum_name).DEFAULT


def maybe_adapt_resnet_for_small_images(model_name: str, model: nn.Module, input_size: int, pretrained: bool):
    if model_name in RESNET_STYLE_MODELS and input_size == 32 and not pretrained:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def _new_like_linear(layer: nn.Linear, out_dim: int) -> nn.Linear:
    return nn.Linear(layer.in_features, out_dim, bias=(layer.bias is not None))


def _new_like_conv2d(layer: nn.Conv2d, out_dim: int) -> nn.Conv2d:
    return nn.Conv2d(
        layer.in_channels,
        out_dim,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=(layer.bias is not None),
        padding_mode=layer.padding_mode,
    )


def _replace_last_linear_or_conv(module: nn.Module, out_dim: int, allow_conv: bool = True) -> bool:
    for name, child in reversed(list(module.named_children())):
        if isinstance(child, nn.Linear):
            module._modules[name] = _new_like_linear(child, out_dim)
            return True
        if allow_conv and isinstance(child, nn.Conv2d):
            module._modules[name] = _new_like_conv2d(child, out_dim)
            return True
        if _replace_last_linear_or_conv(child, out_dim, allow_conv=allow_conv):
            return True
    return False


def _replace_named_head(model: nn.Module, attr: str, out_dim: int) -> bool:
    if not hasattr(model, attr):
        return False
    head = getattr(model, attr)
    if isinstance(head, nn.Linear):
        setattr(model, attr, _new_like_linear(head, out_dim))
        return True
    if isinstance(head, nn.Conv2d):
        setattr(model, attr, _new_like_conv2d(head, out_dim))
        return True
    if isinstance(head, nn.Sequential):
        return _replace_last_linear_or_conv(head, out_dim, allow_conv=True)
    return False


def replace_prediction_head(model_name: str, model: nn.Module, out_dim: int) -> nn.Module:
    """Replace the classifier/regression head for common torchvision models.

    Handles fc/classifier/head/heads attributes plus a recursive fallback.  This
    is intentionally task-agnostic: classification and regression both need the
    last prediction dimension changed to num_outputs.
    """

    # ResNet / RegNet / GoogLeNet / Inception-style
    if _replace_named_head(model, "fc", out_dim):
        if hasattr(model, "AuxLogits") and getattr(model, "AuxLogits") is not None:
            aux = getattr(model, "AuxLogits")
            _replace_named_head(aux, "fc", out_dim)
        return model

    # DenseNet / VGG / AlexNet / MobileNet / EfficientNet / ConvNeXt / SqueezeNet
    if _replace_named_head(model, "classifier", out_dim):
        if hasattr(model, "num_classes"):
            setattr(model, "num_classes", out_dim)
        return model

    # ViT uses model.heads.head.  Some other models use model.head.
    if hasattr(model, "heads") and _replace_last_linear_or_conv(getattr(model, "heads"), out_dim, allow_conv=True):
        return model
    if _replace_named_head(model, "head", out_dim):
        return model

    if _replace_last_linear_or_conv(model, out_dim, allow_conv=False):
        return model
    raise ValueError(f"Could not find a replaceable prediction head for model: {model_name}")


def _replace_lraspp_head(classifier: nn.Module, num_classes: int) -> bool:
    changed = False
    for attr in ("low_classifier", "high_classifier"):
        layer = getattr(classifier, attr, None)
        if isinstance(layer, nn.Conv2d):
            setattr(classifier, attr, _new_like_conv2d(layer, num_classes))
            changed = True
    return changed


def _replace_segmentation_classifier(classifier: nn.Module, num_classes: int) -> nn.Module:
    if _replace_lraspp_head(classifier, num_classes):
        return classifier
    if isinstance(classifier, nn.Conv2d):
        return _new_like_conv2d(classifier, num_classes)
    if isinstance(classifier, nn.Sequential):
        if _replace_last_linear_or_conv(classifier, num_classes, allow_conv=True):
            return classifier
    if _replace_last_linear_or_conv(classifier, num_classes, allow_conv=True):
        return classifier
    raise ValueError("Could not replace segmentation classifier head")


def _build_builtin_model(model_name: str, num_outputs: int, input_size: int, task_type: str) -> nn.Module | None:
    if model_name == "tiny_cnn":
        return TinyCNN(num_outputs)
    if model_name == "tiny_mlp":
        return TinyMLP(num_outputs)
    if model_name == "small_cnn":
        return SmallCNN(num_outputs)
    if model_name == "depthwise_cnn":
        return DepthwiseSeparableCNN(num_outputs)
    if model_name == "small_resnet":
        return SmallResNet(num_outputs, width=32, blocks_per_stage=2)
    if model_name == "wide_small_resnet":
        return SmallResNet(num_outputs, width=48, blocks_per_stage=3)
    if model_name == "mini_vit":
        patch_size = 4 if input_size <= 64 else 8
        return MiniViT(num_outputs, image_size=input_size, patch_size=patch_size)
    if task_type == "segmentation":
        if model_name == "tiny_unet":
            return TinyUNet(num_outputs)
        if model_name == "unet_small":
            return UNetSmall(num_outputs)
        if model_name == "fcn_lite":
            return FCNLite(num_outputs)
        if model_name == "deeplab_lite":
            return DeepLabLite(num_outputs)
    return None


def _build_torchvision_segmentation_model(models, model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    if model_name not in TORCHVISION_SEGMENTATION_SPECS:
        raise ValueError(
            f"Unsupported segmentation model: {model_name}. "
            f"Use one of {list_supported_models('segmentation')}"
        )
    builder_name, weights_enum = TORCHVISION_SEGMENTATION_SPECS[model_name]
    builder = getattr(models.segmentation, builder_name)
    weights = _get_weights(models.segmentation, weights_enum, pretrained)
    if pretrained:
        model = builder(weights=weights)
    else:
        model = builder(weights=None, weights_backbone=None)
    if hasattr(model, "classifier"):
        model.classifier = _replace_segmentation_classifier(model.classifier, num_classes)
    if getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier = _replace_segmentation_classifier(model.aux_classifier, num_classes)
    return model


def _build_torchvision_classification_model(models, model_name: str, num_outputs: int, pretrained: bool, input_size: int) -> nn.Module:
    if model_name not in TORCHVISION_CLASSIFICATION_SPECS:
        raise ValueError(
            f"Unsupported classification/regression model: {model_name}. "
            f"Use one of {list_supported_models('classification')}"
        )
    builder_name, weights_enum = TORCHVISION_CLASSIFICATION_SPECS[model_name]
    builder = getattr(models, builder_name)
    weights = _get_weights(models, weights_enum, pretrained)
    kwargs = {"weights": weights}
    # Avoid auxiliary outputs during training/evaluation when building from scratch.
    if not pretrained and builder_name in {"googlenet", "inception_v3"}:
        kwargs["aux_logits"] = False
    if not pretrained and builder_name == "inception_v3":
        kwargs["init_weights"] = True
    model = builder(**kwargs)
    model = maybe_adapt_resnet_for_small_images(model_name, model, input_size, pretrained)
    return replace_prediction_head(model_name, model, num_outputs)


def build_model(model_name: str, num_outputs: int, pretrained: bool, input_size: int, task_type: str = "classification") -> nn.Module:
    model_name = model_name.lower()
    task_type = task_type.lower()
    if task_type not in {"classification", "regression", "segmentation"}:
        raise ValueError("task_type must be classification, regression, or segmentation")

    built_in = _build_builtin_model(model_name, num_outputs, input_size, task_type)
    if built_in is not None:
        return built_in

    models = _import_torchvision_models()
    if task_type == "segmentation":
        return _build_torchvision_segmentation_model(models, model_name, num_outputs, pretrained)
    return _build_torchvision_classification_model(models, model_name, num_outputs, pretrained, input_size)
