from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Sample:
    """保存一张图像和其像素级标注的路径。"""

    image_path: Path
    label_path: Path


def find_dataset_dir(root: str | Path) -> Path:
    """定位 Stanford Background Dataset 根目录，兼容直接解压或多包一层目录的情况。"""
    root_path = Path(root)
    candidates = [root_path, root_path / "iccv09Data"]
    for child in root_path.glob("*"):
        if child.is_dir():
            candidates.append(child)
            candidates.append(child / "iccv09Data")
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    raise FileNotFoundError(
        f"未找到数据集目录，请确认 {root_path} 下存在 images/ 和 labels/。"
    )


def discover_samples(root: str | Path) -> list[Sample]:
    """扫描图像和 .regions.txt 标签，按文件名配对样本。"""
    dataset_dir = find_dataset_dir(root)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    samples: list[Sample] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.regions.txt"
        if label_path.exists():
            samples.append(Sample(image_path=image_path, label_path=label_path))
    if not samples:
        raise FileNotFoundError("没有找到可配对的图像和 .regions.txt 标签文件。")
    return samples


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """按固定随机种子划分训练、验证和测试集，保证报告中的结果可复现。"""
    shuffled = samples.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


class StanfordBackgroundDataset(Dataset):
    """Stanford Background Dataset 的 PyTorch 数据集封装。

    标签文件是文本矩阵，每个像素位置保存语义类别编号。数据集中可能出现
    unknown 或无效像素，本实现统一映射到 ignore_index，使其不参与损失和指标。
    """

    def __init__(
        self,
        samples: list[Sample],
        image_size: tuple[int, int],
        num_classes: int = 8,
        ignore_index: int = 255,
        training: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = self._read_mask(sample.label_path)

        if self.training:
            image, mask = self._augment(image, mask)

        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        mask_img = Image.fromarray(mask.astype(np.uint8), mode="L")
        mask = np.array(mask_img.resize((self.image_size[1], self.image_size[0]), Image.NEAREST))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor

    def _read_mask(self, path: Path) -> np.ndarray:
        """读取文本标签，并自动处理 1-based 标签和 unknown 像素。"""
        mask = np.loadtxt(path, dtype=np.int64)
        mask = np.atleast_2d(mask)
        invalid = mask < 0
        valid = mask[~invalid]
        if valid.size > 0 and valid.min() >= 1 and valid.max() <= self.num_classes:
            mask = mask - 1
        invalid |= (mask < 0) | (mask >= self.num_classes)
        mask = mask.astype(np.int64)
        mask[invalid] = self.ignore_index
        return mask

    def _augment(self, image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        """训练阶段的轻量数据增强，保持图像和标签的空间变换一致。"""
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask).copy()
        if random.random() < 0.8:
            brightness = 0.85 + random.random() * 0.3
            contrast = 0.85 + random.random() * 0.3
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
        return image, mask
