#!/bin/bash
# 说明：
# 1. 该脚本按顺序训练 CE、Dice、CE+Dice 三种配置。
# 2. 资源参数集中在脚本顶部，便于根据集群实际情况调整。
# 3. 所有启动命令直接调用项目本地 .venv 中的 Python，不依赖激活环境。

#SBATCH --job-name=unet_sbd
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
mkdir -p "$ROOT_DIR/logs/slurm"

cd "$ROOT_DIR"

"$PYTHON_BIN" src/train.py --config configs/unet_ce.yaml
"$PYTHON_BIN" src/train.py --config configs/unet_dice.yaml
"$PYTHON_BIN" src/train.py --config configs/unet_ce_dice.yaml

"$PYTHON_BIN" src/evaluate.py --config configs/unet_ce_dice.yaml --checkpoint checkpoints/unet_ce_dice_best.pt
"$PYTHON_BIN" scripts/make_report.py --results runs --output reports/experiment_report.pdf
