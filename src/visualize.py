from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.utils import colorize_mask


def save_curves(metrics_csv: str | Path, output_dir: str | Path) -> None:
    """根据训练 CSV 生成 loss、Accuracy 和 mIoU 曲线，供报告直接引用。"""
    rows: list[dict[str, float]] = []
    with Path(metrics_csv).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if k != "epoch"})
            rows[-1]["epoch"] = float(row["epoch"])
    if not rows:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    plots = [
        ("loss", "Loss", ["train_loss", "val_loss"]),
        ("accuracy", "Pixel Accuracy", ["train_pixel_accuracy", "val_pixel_accuracy"]),
        ("miou", "Mean IoU", ["train_mean_iou", "val_mean_iou"]),
    ]
    for filename, title, keys in plots:
        plt.figure(figsize=(7, 4))
        for key in keys:
            plt.plot(epochs, [row[key] for row in rows], marker="o", label=key)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f"{filename}.png", dpi=160)
        plt.close()


def save_prediction_grid(
    image_tensor: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    output_path: str | Path,
    ignore_index: int = 255,
) -> None:
    """保存原图、真实标签和预测标签的三联图，用于报告展示模型效果。"""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = image_tensor.detach().cpu().numpy()
    image = np.clip((image * std + mean).transpose(1, 2, 0), 0, 1)
    target_mask = target.detach().cpu().numpy()
    pred_mask = prediction.detach().cpu().numpy()

    panels = [
        ("Image", (image * 255).astype(np.uint8)),
        ("Ground Truth", colorize_mask(target_mask, ignore_index)),
        ("Prediction", colorize_mask(pred_mask, ignore_index)),
    ]
    plt.figure(figsize=(10, 4))
    for index, (title, panel) in enumerate(panels, start=1):
        plt.subplot(1, 3, index)
        plt.imshow(panel)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def save_mask_png(mask: np.ndarray, output_path: str | Path, ignore_index: int = 255) -> None:
    """把单张 mask 保存为彩色 PNG，便于人工快速检查类别区域。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize_mask(mask, ignore_index)).save(output)
