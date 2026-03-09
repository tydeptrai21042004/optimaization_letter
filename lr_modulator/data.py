from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from .config import ExperimentConfig


GRAYSCALE_DATASETS = {
    "mnist",
    "emnist_digits",
    "usps",
    "kmnist",
    "fashionmnist",
}

DATASET_INFO = {
    "cifar10": {
        "num_classes": 10,
        "is_small": True,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "num_classes": 100,
        "is_small": True,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "svhn": {
        "num_classes": 10,
        "is_small": True,
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
    },

    # Added grayscale datasets
    "mnist": {
        "num_classes": 10,
        "is_small": True,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
    "emnist_digits": {
        "num_classes": 10,
        "is_small": True,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
    "usps": {
        "num_classes": 10,
        "is_small": True,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
    "kmnist": {
        "num_classes": 10,
        "is_small": True,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },
    "fashionmnist": {
        "num_classes": 10,
        "is_small": True,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
    },

    "flowers102": {
        "num_classes": 102,
        "is_small": False,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "oxfordiiitpet": {
        "num_classes": 37,
        "is_small": False,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "food101": {
        "num_classes": 101,
        "is_small": False,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}


def _import_torchvision():
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "torchvision could not be imported. In Kaggle, install a compatible torch/torchvision pair. "
            f"Original error: {exc}"
        ) from exc
    return datasets, transforms


def svhn_target_transform(y: int) -> int:
    return int(y) % 10


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def recommended_input_size(model_name: str, dataset_name: str, pretrained: bool) -> int:
    is_small = DATASET_INFO[dataset_name]["is_small"]
    model_name = model_name.lower()

    if not is_small:
        return 224

    if pretrained:
        return 224

    # Small-image friendly CNNs
    if model_name in {
        "resnet18",
        "resnet34",
        "resnet50",
        "resnext50_32x4d",
        "wide_resnet50_2",
        "densenet121",
        "googlenet",
        "shufflenet_v2_x1_0",
        "shufflenet_v2_x0_5",
        "regnet_y_400mf",
        "regnet_x_400mf",
    }:
        return 32

    # Usually okay with a bit larger small-image size
    if model_name in {
        "mobilenet_v2",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "mnasnet0_5",
        "mnasnet1_0",
        "efficientnet_b0",
        "efficientnet_v2_s",
    }:
        return 96

    # Transformer / larger modern backbones
    if model_name in {
        "vit_b_16",
        "swin_t",
        "convnext_tiny",
        "convnext_small",
        "alexnet",
        "vgg11",
        "vgg16",
    }:
        return 224

    return 224


def build_transforms(dataset: str, input_size: int):
    _, transforms = _import_torchvision()
    mean = DATASET_INFO[dataset]["mean"]
    std = DATASET_INFO[dataset]["std"]

    # Special handling for grayscale datasets:
    # convert to 3 channels so RGB backbones can be reused unchanged
    if dataset in GRAYSCALE_DATASETS:
        train_tf = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return train_tf, eval_tf

    if input_size == 32:
        aug = [transforms.RandomCrop(32, padding=4)]
        if dataset != "svhn":
            aug.append(transforms.RandomHorizontalFlip())
        train_tf = transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize(mean, std)])
        eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.Resize(int(round(input_size * 1.14))),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return train_tf, eval_tf


def make_split_indices(n_total: int, val_ratio: float, seed: int):
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    tr, va = random_split(range(n_total), [n_train, n_val], generator=gen)
    return tr.indices, va.indices


def build_datasets(
    dataset: str,
    root: str,
    input_size: int,
    val_ratio: float,
    seed: int,
    download: bool,
):
    datasets, _ = _import_torchvision()
    train_tf, eval_tf = build_transforms(dataset, input_size)

    if dataset == "cifar10":
        full_tr = datasets.CIFAR10(root=root, train=True, transform=train_tf, download=download)
        full_ev = datasets.CIFAR10(root=root, train=True, transform=eval_tf, download=download)
        te = datasets.CIFAR10(root=root, train=False, transform=eval_tf, download=download)
        tr_idx, va_idx = make_split_indices(len(full_tr), val_ratio, seed)
        return Subset(full_tr, tr_idx), Subset(full_ev, va_idx), te

    if dataset == "cifar100":
        full_tr = datasets.CIFAR100(root=root, train=True, transform=train_tf, download=download)
        full_ev = datasets.CIFAR100(root=root, train=True, transform=eval_tf, download=download)
        te = datasets.CIFAR100(root=root, train=False, transform=eval_tf, download=download)
        tr_idx, va_idx = make_split_indices(len(full_tr), val_ratio, seed)
        return Subset(full_tr, tr_idx), Subset(full_ev, va_idx), te

    if dataset == "svhn":
        full_tr = datasets.SVHN(
            root=root,
            split="train",
            transform=train_tf,
            target_transform=svhn_target_transform,
            download=download,
        )
        full_ev = datasets.SVHN(
            root=root,
            split="train",
            transform=eval_tf,
            target_transform=svhn_target_transform,
            download=download,
        )
        te = datasets.SVHN(
            root=root,
            split="test",
            transform=eval_tf,
            target_transform=svhn_target_transform,
            download=download,
        )
        tr_idx, va_idx = make_split_indices(len(full_tr), val_ratio, seed)
        return Subset(full_tr, tr_idx), Subset(full_ev, va_idx), te

    if dataset == "mnist":
        full_tr = datasets.MNIST(root=root, train=True, transform=train_tf, download=download)
        full_ev = datasets.MNIST(root=root, train=True, transform=eval_tf, download=download)
        te = datasets.MNIST(root=root, train=False, transform=eval_tf, download=download)
        tr_idx, va_idx = make_split_indices(len(full_tr), val_ratio, seed)
        return Subset(full_tr, tr_idx), Subset(full_ev, va_idx), te

    if dataset == "emnist_digits":
        full_tr = datasets.EMNIST(
            root=root,
            split="digits",
            train=True,
            transform=train_tf,
            download=download,
        )
        full_ev = datasets.EMNIST(
            root=root,
            split="digits",
            train=True,
            transform=eval_tf,
            download=download,
        )
        te = datasets.EMNIST(
            root=root,
            split="digits",
            train=False,
            transform=eval_tf,
            download=download,
        )
        tr_idx, va_idx = make_split_indices(len(full_tr), val_ratio, seed)
        return Subset(full_tr, tr_idx), Subset(full_ev, va_idx), te

    if dataset == "usps":
        full_tr = datasets.USPS(root=root, train=True, transform=train_tf, download=download)
        full_ev = datasets.USPS(root=root, train=True, transform=eval_tf, download=download)
        te = datasets.USPS(root=root, train=False, transform=eval_tf, download=download)
        tr_idx, va_idx = make_split_indices(len(full_tr), val_ratio, seed)
        return Subset(full_tr, tr_idx), Subset(full_ev, va_idx), te

    if dataset == "kmnist":
        full_tr = datasets.KMNIST(root=root, train=True, transform=train_tf, download=download)
        full_ev = datasets.KMNIST(root=root, train=True, transform=eval_tf, download=download)
        te = datasets.KMNIST(root=root, train=False, transform=eval_tf, download=download)
        tr_idx, va_idx = make_split_indices(len(full_tr), val_ratio, seed)
        return Subset(full_tr, tr_idx), Subset(full_ev, va_idx), te

    if dataset == "fashionmnist":
        full_tr = datasets.FashionMNIST(root=root, train=True, transform=train_tf, download=download)
        full_ev = datasets.FashionMNIST(root=root, train=True, transform=eval_tf, download=download)
        te = datasets.FashionMNIST(root=root, train=False, transform=eval_tf, download=download)
        tr_idx, va_idx = make_split_indices(len(full_tr), val_ratio, seed)
        return Subset(full_tr, tr_idx), Subset(full_ev, va_idx), te

    if dataset == "flowers102":
        tr = datasets.Flowers102(root=root, split="train", transform=train_tf, download=download)
        va = datasets.Flowers102(root=root, split="val", transform=eval_tf, download=download)
        te = datasets.Flowers102(root=root, split="test", transform=eval_tf, download=download)
        return tr, va, te

    if dataset == "oxfordiiitpet":
        full_tr = datasets.OxfordIIITPet(
            root=root,
            split="trainval",
            target_types="category",
            transform=train_tf,
            download=download,
        )
        full_ev = datasets.OxfordIIITPet(
            root=root,
            split="trainval",
            target_types="category",
            transform=eval_tf,
            download=download,
        )
        te = datasets.OxfordIIITPet(
            root=root,
            split="test",
            target_types="category",
            transform=eval_tf,
            download=download,
        )
        tr_idx, va_idx = make_split_indices(len(full_tr), val_ratio, seed)
        return Subset(full_tr, tr_idx), Subset(full_ev, va_idx), te

    if dataset == "food101":
        full_tr = datasets.Food101(root=root, split="train", transform=train_tf, download=download)
        full_ev = datasets.Food101(root=root, split="train", transform=eval_tf, download=download)
        te = datasets.Food101(root=root, split="test", transform=eval_tf, download=download)
        tr_idx, va_idx = make_split_indices(len(full_tr), val_ratio, seed)
        return Subset(full_tr, tr_idx), Subset(full_ev, va_idx), te

    raise ValueError(f"Unsupported dataset: {dataset}")


def build_loaders(
    config: ExperimentConfig,
    device: torch.device,
    dataset: str,
    input_size: int,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    tr_ds, va_ds, te_ds = build_datasets(
        dataset=dataset,
        root=config.data_root,
        input_size=input_size,
        val_ratio=config.val_ratio,
        seed=seed,
        download=config.download,
    )

    kwargs = dict(
        batch_size=batch_size,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
        worker_init_fn=seed_worker,
    )

    train_gen = make_loader_generator(seed)
    eval_gen = make_loader_generator(seed + 1)

    return (
        DataLoader(tr_ds, shuffle=True, generator=train_gen, **kwargs),
        DataLoader(va_ds, shuffle=False, generator=eval_gen, **kwargs),
        DataLoader(te_ds, shuffle=False, generator=eval_gen, **kwargs),
        DATASET_INFO[dataset]["num_classes"],
    )