from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import StanfordBackgroundDataset, discover_samples, split_samples
from src.losses import SegmentationLoss
from src.metrics import SegmentationMetrics
from src.model import UNet
from src.utils import append_csv, load_config, resolve_device, save_json, set_seed
from src.visualize import save_curves


def build_loaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """根据配置构建训练和验证 DataLoader。"""
    data_cfg = config["data"]
    samples = discover_samples(data_cfg["root"])
    train_samples, val_samples, _ = split_samples(
        samples,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=config["seed"],
    )
    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_val_samples")
    if max_train:
        train_samples = train_samples[: int(max_train)]
    if max_val:
        val_samples = val_samples[: int(max_val)]

    image_size = tuple(data_cfg["image_size"])
    train_dataset = StanfordBackgroundDataset(
        train_samples,
        image_size=image_size,
        num_classes=data_cfg["num_classes"],
        ignore_index=data_cfg["ignore_index"],
        training=True,
    )
    val_dataset = StanfordBackgroundDataset(
        val_samples,
        image_size=image_size,
        num_classes=data_cfg["num_classes"],
        ignore_index=data_cfg["ignore_index"],
        training=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    """执行一个训练或验证 epoch；传入 optimizer 时为训练模式，否则为验证模式。"""
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    total_loss = 0.0
    total_items = 0
    iterator = tqdm(loader, desc="train" if training else "val", leave=False)
    for images, masks in iterator:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        metrics.update(logits.detach(), masks.detach())
        iterator.set_postfix(loss=total_loss / max(total_items, 1))

    computed = metrics.compute()
    return {
        "loss": total_loss / max(total_items, 1),
        "pixel_accuracy": float(computed["pixel_accuracy"]),
        "mean_iou": float(computed["mean_iou"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="训练从零实现的 U-Net 语义分割模型。")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径。")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epoch 数，便于 smoke test。")
    parser.add_argument("--no-swanlab", action="store_true", help="关闭 SwanLab，仅保存本地日志。")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    set_seed(int(config["seed"]))

    device = resolve_device(config["train"]["device"])
    data_cfg = config["data"]
    train_loader, val_loader = build_loaders(config)
    model = UNet(num_classes=data_cfg["num_classes"]).to(device)
    criterion = SegmentationLoss(
        config["train"]["loss"],
        num_classes=data_cfg["num_classes"],
        ignore_index=data_cfg["ignore_index"],
        ce_weight=config["train"]["ce_weight"],
        dice_weight=config["train"]["dice_weight"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if config["train"].get("amp", True) and device.type == "cuda"
        else None
    )

    run_dir = Path(config["logging"]["run_dir"])
    checkpoint_path = Path(config["logging"]["checkpoint_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "config.yaml")
    save_json(config, run_dir / "resolved_config.json")
    metrics_csv = run_dir / "metrics.csv"
    best_miou = -1.0

    swanlab_run = None
    if config["logging"].get("use_swanlab", False) and not args.no_swanlab:
        try:
            import swanlab

            swanlab_run = swanlab.init(
                project="stanford-background-unet",
                experiment_name=config["experiment_name"],
                config=config,
            )
        except Exception as exc:
            print(f"SwanLab 初始化失败，继续使用本地日志：{exc}")

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            num_classes=data_cfg["num_classes"],
            ignore_index=data_cfg["ignore_index"],
            optimizer=optimizer,
            scaler=scaler,
            use_amp=config["train"].get("amp", True),
        )
        val_stats = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes=data_cfg["num_classes"],
            ignore_index=data_cfg["ignore_index"],
            use_amp=config["train"].get("amp", True),
        )
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_pixel_accuracy": train_stats["pixel_accuracy"],
            "val_pixel_accuracy": val_stats["pixel_accuracy"],
            "train_mean_iou": train_stats["mean_iou"],
            "val_mean_iou": val_stats["mean_iou"],
        }
        append_csv(metrics_csv, row)
        if swanlab_run is not None:
            try:
                import swanlab

                swanlab.log(row, step=epoch)
            except Exception as exc:
                print(f"SwanLab 记录失败，已保留本地日志：{exc}")

        if val_stats["mean_iou"] > best_miou:
            best_miou = val_stats["mean_iou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_mean_iou": best_miou,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch}: train_loss={train_stats['loss']:.4f}, "
            f"val_loss={val_stats['loss']:.4f}, val_mIoU={val_stats['mean_iou']:.4f}"
        )

    save_curves(metrics_csv, run_dir)
    save_json({"best_val_mean_iou": best_miou}, run_dir / "summary.json")
    if swanlab_run is not None:
        try:
            import swanlab

            swanlab.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
