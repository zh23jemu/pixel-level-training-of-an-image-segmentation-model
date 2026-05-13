#!/bin/bash
# 说明：
# 1. 该脚本按顺序训练 CE、Dice、CE+Dice 三种配置。
# 2. GPU 作业按当前集群策略优先使用 aws 分区。
# 3. 所有启动命令直接调用项目本地 .venv 中的 Python，不依赖激活环境。

#SBATCH --job-name=unet_sbd
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00

set -e

# Slurm 可能会把脚本复制到计算节点的 spool 目录后再执行，此时 "$0" 不再指向项目内脚本。
# 因此优先使用提交作业时的目录，确保日志、权重和训练结果都写回共享项目目录，而不是计算节点本地路径。
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
mkdir -p "$ROOT_DIR/logs/slurm"

cd "$ROOT_DIR"

"$PYTHON_BIN" src/train.py --config configs/unet_ce.yaml
"$PYTHON_BIN" src/train.py --config configs/unet_dice.yaml
"$PYTHON_BIN" src/train.py --config configs/unet_ce_dice.yaml
