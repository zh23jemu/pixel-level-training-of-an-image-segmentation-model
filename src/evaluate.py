from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import StanfordBackgroundDataset, discover_samples, split_samples
from src.metrics import SegmentationMetrics
from src.model import UNet
from src.utils import CLASS_NAMES, load_config, resolve_device, save_json
from src.visualize import save_prediction_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 U-Net 语义分割模型并保存可视化结果。")
    parser.add_argument("--config", required=True, help="训练时使用的 YAML 配置。")
    parser.add_argument("--checkpoint", required=True, help="待评估的模型权重。")
    parser.add_argument("--output-dir", default=None, help="测试结果输出目录，默认写入 run_dir/test。")
    parser.add_argument("--max-visualizations", type=int, default=8, help="最多保存多少张预测可视化。")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    device = resolve_device(config["train"]["device"])
    output_dir = Path(args.output_dir or (Path(config["logging"]["run_dir"]) / "test"))
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(data_cfg["root"])
    _, _, test_samples = split_samples(
        samples,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=config["seed"],
    )
    max_test = data_cfg.get("max_test_samples")
    if max_test:
        test_samples = test_samples[: int(max_test)]

    dataset = StanfordBackgroundDataset(
        test_samples,
        image_size=tuple(data_cfg["image_size"]),
        num_classes=data_cfg["num_classes"],
        ignore_index=data_cfg["ignore_index"],
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(num_classes=data_cfg["num_classes"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metrics = SegmentationMetrics(
        num_classes=data_cfg["num_classes"],
        ignore_index=data_cfg["ignore_index"],
    )
    saved = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="test"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            metrics.update(logits, masks)
            preds = torch.argmax(logits, dim=1)
            for i in range(images.size(0)):
                if saved >= args.max_visualizations:
                    break
                save_prediction_grid(
                    images[i].cpu(),
                    masks[i].cpu(),
                    preds[i].cpu(),
                    output_dir / f"prediction_{saved:02d}.png",
                    ignore_index=data_cfg["ignore_index"],
                )
                saved += 1

    result = metrics.compute()
    per_class = {
        CLASS_NAMES[i]: result["per_class_iou"][i]
        for i in range(min(len(CLASS_NAMES), data_cfg["num_classes"]))
    }
    save_json(
        {
            "checkpoint": str(args.checkpoint),
            "pixel_accuracy": result["pixel_accuracy"],
            "mean_iou": result["mean_iou"],
            "per_class_iou": per_class,
        },
        output_dir / "metrics.json",
    )
    print(f"Test pixel accuracy: {result['pixel_accuracy']:.4f}")
    print(f"Test mIoU: {result['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
