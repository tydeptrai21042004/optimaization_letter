from __future__ import annotations

import torch
import torch.nn as nn

torch.set_num_threads(1)

from lr_modulator.data import recommended_input_size
from lr_modulator.model_zoo import build_model, list_supported_models, replace_prediction_head


def test_builtin_classification_and_regression_backbones_forward() -> None:
    names = ["tiny_mlp", "tiny_cnn", "small_cnn", "depthwise_cnn", "small_resnet", "wide_small_resnet", "mini_vit"]
    for name in names:
        input_size = recommended_input_size(name, "synthetic_classification", pretrained=False)
        x = torch.randn(2, 3, input_size, input_size)
        cls_model = build_model(name, num_outputs=3, pretrained=False, input_size=input_size, task_type="classification")
        reg_model = build_model(name, num_outputs=1, pretrained=False, input_size=input_size, task_type="regression")
        cls_out = cls_model(x)
        reg_out = reg_model(x)
        assert cls_out.shape == (2, 3), f"{name} classification output shape mismatch"
        assert reg_out.shape == (2, 1), f"{name} regression output shape mismatch"
        assert torch.isfinite(cls_out).all()
        assert torch.isfinite(reg_out).all()


def test_builtin_segmentation_architectures_forward() -> None:
    names = ["tiny_unet", "unet_small", "fcn_lite", "deeplab_lite"]
    for name in names:
        input_size = recommended_input_size(name, "synthetic_segmentation", pretrained=False)
        x = torch.randn(2, 3, input_size, input_size)
        model = build_model(name, num_outputs=2, pretrained=False, input_size=input_size, task_type="segmentation")
        out = model(x)
        assert out.shape == (2, 2, input_size, input_size), f"{name} segmentation output shape mismatch"
        assert torch.isfinite(out).all()


class FakeResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 1000)


class FakeDenseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(12, 1000)


class FakeSequentialClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(16, 1000))


class FakeSqueezeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(32, 1000, 1), nn.ReLU(inplace=True))


class FakeViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.Sequential(nn.LayerNorm(20), nn.Linear(20, 1000))


def test_head_replacement_patterns_for_torchvision_like_models() -> None:
    fake_models = [
        (FakeResNet(), "fc", nn.Linear),
        (FakeDenseNet(), "classifier", nn.Linear),
        (FakeSequentialClassifier(), "classifier", nn.Sequential),
        (FakeSqueezeNet(), "classifier", nn.Sequential),
        (FakeViT(), "heads", nn.Sequential),
    ]
    for model, _, _ in fake_models:
        updated = replace_prediction_head("fake", model, out_dim=7)
        # find the final replaceable layer and check its output dimension
        final_linear_or_conv = None
        for module in updated.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                final_linear_or_conv = module
        assert final_linear_or_conv is not None
        if isinstance(final_linear_or_conv, nn.Linear):
            assert final_linear_or_conv.out_features == 7
        else:
            assert final_linear_or_conv.out_channels == 7


def test_supported_model_lists_include_new_backbones() -> None:
    cls = set(list_supported_models("classification"))
    seg = set(list_supported_models("segmentation"))
    assert {"small_resnet", "depthwise_cnn", "mini_vit", "efficientnet_b7", "convnext_base", "swin_v2_t"}.issubset(cls)
    assert {"tiny_unet", "unet_small", "fcn_lite", "deeplab_lite", "deeplabv3_mobilenet_v3_large"}.issubset(seg)
