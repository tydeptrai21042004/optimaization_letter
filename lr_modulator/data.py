from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .config import ExperimentConfig

GRAYSCALE_DATASETS = {"mnist", "emnist_digits", "usps", "kmnist", "fashionmnist"}
SEGMENTATION_DATASETS = {"voc_segmentation", "pet_segmentation", "synthetic_segmentation"}

DATASET_INFO: Dict[str, Dict[str, object]] = {
    "cifar10": {"task_type": "classification", "num_classes": 10, "is_small": True, "mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010)},
    "cifar100": {"task_type": "classification", "num_classes": 100, "is_small": True, "mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
    "svhn": {"task_type": "classification", "num_classes": 10, "is_small": True, "mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970)},
    "mnist": {"task_type": "classification", "num_classes": 10, "is_small": True, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "emnist_digits": {"task_type": "classification", "num_classes": 10, "is_small": True, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "usps": {"task_type": "classification", "num_classes": 10, "is_small": True, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "kmnist": {"task_type": "classification", "num_classes": 10, "is_small": True, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "fashionmnist": {"task_type": "classification", "num_classes": 10, "is_small": True, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "stl10": {"task_type": "classification", "num_classes": 10, "is_small": False, "mean": (0.4467, 0.4398, 0.4066), "std": (0.2603, 0.2566, 0.2713)},
    "gtsrb": {"task_type": "classification", "num_classes": 43, "is_small": False, "mean": (0.3403, 0.3121, 0.3214), "std": (0.2724, 0.2608, 0.2669)},
    "dtd": {"task_type": "classification", "num_classes": 47, "is_small": False, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "eurosat": {"task_type": "classification", "num_classes": 10, "is_small": False, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "country211": {"task_type": "classification", "num_classes": 211, "is_small": False, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "caltech101": {"task_type": "classification", "num_classes": 101, "is_small": False, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "caltech256": {"task_type": "classification", "num_classes": 257, "is_small": False, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "flowers102": {"task_type": "classification", "num_classes": 102, "is_small": False, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "oxfordiiitpet": {"task_type": "classification", "num_classes": 37, "is_small": False, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "food101": {"task_type": "classification", "num_classes": 101, "is_small": False, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "synthetic_classification": {"task_type": "classification", "num_classes": 3, "is_small": True, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "synthetic_regression": {"task_type": "regression", "num_outputs": 1, "is_small": True, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "synthetic_segmentation": {"task_type": "segmentation", "num_classes": 2, "ignore_index": 255, "is_small": True, "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "voc_segmentation": {"task_type": "segmentation", "num_classes": 21, "ignore_index": 255, "is_small": False, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "pet_segmentation": {"task_type": "segmentation", "num_classes": 3, "ignore_index": 255, "is_small": False, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


def _import_torchvision():
    try:
        from torchvision import datasets, transforms
    except Exception as exc:
        raise RuntimeError(f"torchvision could not be imported. Original error: {exc}") from exc
    return datasets, transforms


def dataset_info(dataset: str) -> Dict[str, object]:
    dataset = dataset.lower()
    if dataset not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {sorted(DATASET_INFO)}")
    return DATASET_INFO[dataset]


def task_type_for_dataset(dataset: str) -> str:
    return str(dataset_info(dataset).get("task_type", "classification"))


def num_outputs_for_dataset(dataset: str) -> int:
    info = dataset_info(dataset)
    return int(info.get("num_classes", info.get("num_outputs", 1)))


def svhn_target_transform(y: int) -> int:
    return int(y) % 10


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator().manual_seed(seed)
    return g


def recommended_input_size(model_name: str, dataset_name: str, pretrained: bool) -> int:
    """Choose a safe default input size for each dataset/model pair.

    The rule favors small images for built-in/debug models and CIFAR-style
    ResNets, but keeps ImageNet-style or fixed-position-embedding models at
    their expected resolution.  This prevents silent shape failures when users
    switch backbones from the CLI.
    """
    info = dataset_info(dataset_name)
    model_name = model_name.lower()

    built_in_classification = {
        "tiny_mlp", "tiny_cnn", "small_cnn", "depthwise_cnn",
        "small_resnet", "wide_small_resnet", "mini_vit",
    }
    built_in_segmentation = {"tiny_unet", "unet_small", "fcn_lite", "deeplab_lite"}
    cifar_safe_torchvision = {
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2",
        "densenet121", "densenet161", "densenet169", "densenet201",
        "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_v2_s",
        "convnext_tiny", "regnet_x_400mf", "regnet_y_400mf",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
        "mnasnet0_5", "mnasnet1_0", "squeezenet1_0", "squeezenet1_1",
    }
    imagenet_size_models = {
        "alexnet", "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn",
        "vgg19", "vgg19_bn", "googlenet", "maxvit_t",
        "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
        "swin_t", "swin_s", "swin_b", "swin_v2_t", "swin_v2_s", "swin_v2_b",
    }

    if info.get("task_type") == "segmentation":
        if model_name in built_in_segmentation or dataset_name.lower() == "synthetic_segmentation":
            return 32
        return 224

    if model_name == "inception_v3":
        return 299
    if pretrained or model_name in imagenet_size_models:
        return 224
    if bool(info["is_small"]) and (model_name in built_in_classification or model_name in cifar_safe_torchvision):
        return 32
    if bool(info["is_small"]):
        return 96
    return 224


class SyntheticImageClassificationDataset(Dataset):
    def __init__(self, n: int, image_size: int = 16, num_classes: int = 3, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, image_size, image_size, generator=gen)
        raw = self.x.mean(dim=(1, 2, 3)) + 0.15 * self.x[:, 0].mean(dim=(1, 2))
        bins = torch.linspace(float(raw.min()), float(raw.max()) + 1e-6, num_classes + 1)
        self.y = torch.bucketize(raw, bins[1:-1]).long()
    def __len__(self): return int(self.x.size(0))
    def __getitem__(self, i): return self.x[i], self.y[i]


class SyntheticImageRegressionDataset(Dataset):
    def __init__(self, n: int, image_size: int = 16, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, image_size, image_size, generator=gen)
        self.y = (self.x[:, 0].mean((1, 2)) - 0.5 * self.x[:, 1].mean((1, 2))).unsqueeze(1).float()
    def __len__(self): return int(self.x.size(0))
    def __getitem__(self, i): return self.x[i], self.y[i]


class SyntheticSegmentationDataset(Dataset):
    def __init__(self, n: int, image_size: int = 32, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, image_size, image_size, generator=gen)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, image_size), torch.linspace(-1, 1, image_size), indexing="ij")
        circle = (xx * xx + yy * yy < 0.45).float()
        self.y = torch.stack([(self.x[i, 0] + 0.25 * self.x[i, 1] + circle > 0).long() for i in range(n)])
    def __len__(self): return int(self.x.size(0))
    def __getitem__(self, i): return self.x[i], self.y[i]


class SegmentationPairTransform:
    def __init__(self, dataset: str, input_size: int, train: bool):
        _, transforms = _import_torchvision()
        self.dataset, self.input_size, self.train = dataset, int(input_size), bool(train)
        self.normalize = transforms.Normalize(mean=tuple(dataset_info(dataset)["mean"]), std=tuple(dataset_info(dataset)["std"]))
    def __call__(self, image, target):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as F
        image = F.resize(image, [self.input_size, self.input_size], interpolation=InterpolationMode.BILINEAR)
        target = F.resize(target, [self.input_size, self.input_size], interpolation=InterpolationMode.NEAREST)
        if self.train and random.random() < 0.5:
            image, target = F.hflip(image), F.hflip(target)
        x = self.normalize(F.to_tensor(image))
        y = np.asarray(target, dtype=np.int64)
        if self.dataset == "pet_segmentation":
            y = np.maximum(y - 1, 0)
        return x, torch.as_tensor(y, dtype=torch.long)


class SegmentationDatasetWrapper(Dataset):
    def __init__(self, base: Dataset, transform: SegmentationPairTransform):
        self.base, self.transform = base, transform
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        image, target = self.base[i]
        return self.transform(image, target)


def build_transforms(dataset: str, input_size: int):
    _, transforms = _import_torchvision()
    info = dataset_info(dataset)
    mean, std = info["mean"], info["std"]
    if dataset in GRAYSCALE_DATASETS:
        tf = [transforms.Resize((input_size, input_size)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize(mean, std)]
        return transforms.Compose(tf), transforms.Compose(tf)
    if input_size == 32:
        aug = [transforms.RandomCrop(32, padding=4)]
        if dataset != "svhn": aug.append(transforms.RandomHorizontalFlip())
        return transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize(mean, std)]), transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return (
        transforms.Compose([transforms.RandomResizedCrop(input_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize(int(round(input_size * 1.14))), transforms.CenterCrop(input_size), transforms.ToTensor(), transforms.Normalize(mean, std)]),
    )


def make_split_indices(n_total: int, val_ratio: float, seed: int):
    n_val = min(max(1, int(round(n_total * val_ratio))), max(1, n_total - 1))
    n_train = n_total - n_val
    tr, va = random_split(range(n_total), [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    return tr.indices, va.indices


def _split_train_val(full_train: Dataset, full_eval: Dataset, val_ratio: float, seed: int):
    tr_idx, va_idx = make_split_indices(len(full_train), val_ratio, seed)
    return Subset(full_train, tr_idx), Subset(full_eval, va_idx)


def _build_synthetic_datasets(dataset: str, input_size: int, seed: int):
    if dataset == "synthetic_classification":
        full_tr = SyntheticImageClassificationDataset(48, input_size, 3, seed); full_ev = SyntheticImageClassificationDataset(48, input_size, 3, seed); te = SyntheticImageClassificationDataset(16, input_size, 3, seed + 100)
    elif dataset == "synthetic_regression":
        full_tr = SyntheticImageRegressionDataset(48, input_size, seed); full_ev = SyntheticImageRegressionDataset(48, input_size, seed); te = SyntheticImageRegressionDataset(16, input_size, seed + 100)
    elif dataset == "synthetic_segmentation":
        full_tr = SyntheticSegmentationDataset(8, input_size, seed); full_ev = SyntheticSegmentationDataset(8, input_size, seed); te = SyntheticSegmentationDataset(4, input_size, seed + 100)
    else:
        return None
    tr, va = _split_train_val(full_tr, full_ev, 0.25, seed)
    return tr, va, te


def build_datasets(dataset: str, root: str, input_size: int, val_ratio: float, seed: int, download: bool):
    dataset = dataset.lower(); dataset_info(dataset)
    synthetic = _build_synthetic_datasets(dataset, input_size, seed)
    if synthetic is not None: return synthetic
    datasets, _ = _import_torchvision()
    if dataset in SEGMENTATION_DATASETS:
        train_tf = SegmentationPairTransform(dataset, input_size, True); eval_tf = SegmentationPairTransform(dataset, input_size, False)
        if dataset == "voc_segmentation":
            full_tr = datasets.VOCSegmentation(root=root, year="2012", image_set="train", download=download); full_ev = datasets.VOCSegmentation(root=root, year="2012", image_set="train", download=download); te = datasets.VOCSegmentation(root=root, year="2012", image_set="val", download=download)
        elif dataset == "pet_segmentation":
            full_tr = datasets.OxfordIIITPet(root=root, split="trainval", target_types="segmentation", download=download); full_ev = datasets.OxfordIIITPet(root=root, split="trainval", target_types="segmentation", download=download); te = datasets.OxfordIIITPet(root=root, split="test", target_types="segmentation", download=download)
        tr, va = _split_train_val(SegmentationDatasetWrapper(full_tr, train_tf), SegmentationDatasetWrapper(full_ev, eval_tf), val_ratio, seed)
        return tr, va, SegmentationDatasetWrapper(te, eval_tf)
    train_tf, eval_tf = build_transforms(dataset, input_size)
    def split_pair(cls, train_args, test_args):
        full_tr = cls(**train_args, transform=train_tf, download=download); full_ev = cls(**train_args, transform=eval_tf, download=download); te = cls(**test_args, transform=eval_tf, download=download)
        tr, va = _split_train_val(full_tr, full_ev, val_ratio, seed); return tr, va, te
    if dataset == "cifar10": return split_pair(datasets.CIFAR10, {"root": root, "train": True}, {"root": root, "train": False})
    if dataset == "cifar100": return split_pair(datasets.CIFAR100, {"root": root, "train": True}, {"root": root, "train": False})
    if dataset == "svhn":
        full_tr = datasets.SVHN(root=root, split="train", transform=train_tf, target_transform=svhn_target_transform, download=download); full_ev = datasets.SVHN(root=root, split="train", transform=eval_tf, target_transform=svhn_target_transform, download=download); te = datasets.SVHN(root=root, split="test", transform=eval_tf, target_transform=svhn_target_transform, download=download)
        tr, va = _split_train_val(full_tr, full_ev, val_ratio, seed); return tr, va, te
    if dataset == "mnist": return split_pair(datasets.MNIST, {"root": root, "train": True}, {"root": root, "train": False})
    if dataset == "emnist_digits": return split_pair(datasets.EMNIST, {"root": root, "split": "digits", "train": True}, {"root": root, "split": "digits", "train": False})
    if dataset == "usps": return split_pair(datasets.USPS, {"root": root, "train": True}, {"root": root, "train": False})
    if dataset == "kmnist": return split_pair(datasets.KMNIST, {"root": root, "train": True}, {"root": root, "train": False})
    if dataset == "fashionmnist": return split_pair(datasets.FashionMNIST, {"root": root, "train": True}, {"root": root, "train": False})
    if dataset == "stl10": return split_pair(datasets.STL10, {"root": root, "split": "train"}, {"root": root, "split": "test"})
    if dataset == "gtsrb": return split_pair(datasets.GTSRB, {"root": root, "split": "train"}, {"root": root, "split": "test"})
    if dataset == "dtd": return datasets.DTD(root=root, split="train", transform=train_tf, download=download), datasets.DTD(root=root, split="val", transform=eval_tf, download=download), datasets.DTD(root=root, split="test", transform=eval_tf, download=download)
    if dataset == "flowers102": return datasets.Flowers102(root=root, split="train", transform=train_tf, download=download), datasets.Flowers102(root=root, split="val", transform=eval_tf, download=download), datasets.Flowers102(root=root, split="test", transform=eval_tf, download=download)
    if dataset == "country211": return datasets.Country211(root=root, split="train", transform=train_tf, download=download), datasets.Country211(root=root, split="valid", transform=eval_tf, download=download), datasets.Country211(root=root, split="test", transform=eval_tf, download=download)
    if dataset == "oxfordiiitpet": return split_pair(datasets.OxfordIIITPet, {"root": root, "split": "trainval", "target_types": "category"}, {"root": root, "split": "test", "target_types": "category"})
    if dataset == "food101": return split_pair(datasets.Food101, {"root": root, "split": "train"}, {"root": root, "split": "test"})
    if dataset in {"eurosat", "caltech101", "caltech256"}:
        cls = {"eurosat": datasets.EuroSAT, "caltech101": datasets.Caltech101, "caltech256": datasets.Caltech256}[dataset]
        full_tr = cls(root=root, transform=train_tf, download=download); full_ev = cls(root=root, transform=eval_tf, download=download)
        tr, va = _split_train_val(full_tr, full_ev, val_ratio, seed); return tr, va, va
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_loaders(config: ExperimentConfig, device: torch.device, dataset: str, input_size: int, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, int, str]:
    dataset = dataset.lower()
    tr_ds, va_ds, te_ds = build_datasets(dataset, config.data_root, input_size, config.val_ratio, seed, config.download)
    kwargs = dict(batch_size=batch_size, num_workers=config.num_workers, pin_memory=(device.type == "cuda"), persistent_workers=(config.num_workers > 0), worker_init_fn=seed_worker)
    train_gen, eval_gen = make_loader_generator(seed), make_loader_generator(seed + 1)
    return DataLoader(tr_ds, shuffle=True, generator=train_gen, **kwargs), DataLoader(va_ds, shuffle=False, generator=eval_gen, **kwargs), DataLoader(te_ds, shuffle=False, generator=eval_gen, **kwargs), num_outputs_for_dataset(dataset), task_type_for_dataset(dataset)
