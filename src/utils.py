from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


CLASS_NAMES = [
    "sky",
    "tree",
    "road",
    "grass",
    "water",
    "building",
    "mountain",
    "foreground_object",
]


PALETTE = np.array(
    [
        [128, 180, 255],
        [34, 139, 34],
        [128, 128, 128],
        [124, 252, 0],
        [30, 144, 255],
        [178, 132, 90],
        [160, 160, 160],
        [220, 20, 60],
        [0, 0, 0],
    ],
    dtype=np.uint8,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，并保留为普通字典，便于脚本按字段访问。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """保存 JSON 文件，用于记录配置快照、指标摘要和测试结果。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    """追加写入训练指标；首次写入时自动创建表头。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子，减少实验结果波动。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    """根据配置选择设备；auto 会优先使用 CUDA。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def colorize_mask(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """把类别索引图转换为彩色可视化图，unknown 像素显示为黑色。"""
    safe_mask = mask.copy()
    safe_mask[safe_mask == ignore_index] = len(PALETTE) - 1
    safe_mask = np.clip(safe_mask, 0, len(PALETTE) - 1)
    return PALETTE[safe_mask]
