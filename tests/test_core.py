from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import StanfordBackgroundDataset, discover_samples, find_dataset_dir
from src.losses import DiceLoss
from src.metrics import SegmentationMetrics
from src.model import UNet


def test_dataset_reads_image_and_regions_mask(tmp_path: Path) -> None:
    """验证数据集能正确配对图像与 .regions.txt，并把负数标签映射为 ignore_index。"""
    images = tmp_path / "iccv09Data" / "images"
    labels = tmp_path / "iccv09Data" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(images / "sample.jpg")
    np.savetxt(labels / "sample.regions.txt", np.array([[0, 1, -1, 2, 3]] * 4), fmt="%d")

    samples = discover_samples(tmp_path)
    dataset = StanfordBackgroundDataset(samples, image_size=(8, 10), training=False)
    image, mask = dataset[0]

    assert image.shape == (3, 8, 10)
    assert mask.shape == (8, 10)
    assert 255 in mask


def test_dataset_dir_detection_supports_raw_nested_extract(tmp_path: Path) -> None:
    """验证脚本解压出的 data/raw/iccv09Data 结构可以被自动识别。"""
    dataset_root = tmp_path / "raw" / "iccv09Data"
    (dataset_root / "images").mkdir(parents=True)
    (dataset_root / "labels").mkdir(parents=True)
    assert find_dataset_dir(tmp_path) == dataset_root


def test_unet_keeps_spatial_shape() -> None:
    """验证 U-Net 输出分辨率与输入分辨率一致，满足像素级分割要求。"""
    model = UNet(num_classes=8, base_channels=8)
    x = torch.randn(2, 3, 64, 80)
    y = model(x)
    assert y.shape == (2, 8, 64, 80)


def test_dice_loss_ignores_unknown_pixels() -> None:
    """验证 Dice Loss 遇到 unknown 像素时可以正常反传。"""
    loss_fn = DiceLoss(num_classes=8, ignore_index=255)
    logits = torch.randn(2, 8, 16, 16, requires_grad=True)
    target = torch.randint(0, 8, (2, 16, 16))
    target[:, :2, :2] = 255
    loss = loss_fn(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_metrics_reports_accuracy_and_miou() -> None:
    """验证指标模块能输出 pixel accuracy 和 mIoU。"""
    metrics = SegmentationMetrics(num_classes=2, ignore_index=255)
    logits = torch.tensor([[[[4.0, 0.0]], [[0.0, 4.0]]]])
    target = torch.tensor([[[0, 1]]])
    metrics.update(logits, target)
    result = metrics.compute()
    assert result["pixel_accuracy"] == 1.0
    assert result["mean_iou"] == 1.0
