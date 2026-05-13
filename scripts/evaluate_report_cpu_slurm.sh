#!/bin/bash
# 说明：
# 1. 该脚本用于训练完成后的 CPU 后处理：测试集评估、预测可视化和 PDF 报告生成。
# 2. CPU 作业按当前集群策略优先避开 aws-com，默认使用 defq 分区。
# 3. 所有启动命令直接调用项目本地 .venv 中的 Python，不依赖激活环境。

#SBATCH --job-name=unet_sbd_eval
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -e

# Slurm 可能会把脚本复制到计算节点的 spool 目录后再执行，此时 "$0" 不再指向项目内脚本。
# 因此优先使用提交作业时的目录，确保评估结果和报告都写回共享项目目录，而不是计算节点本地路径。
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
mkdir -p "$ROOT_DIR/logs/slurm"

cd "$ROOT_DIR"

"$PYTHON_BIN" src/evaluate.py --config configs/unet_ce.yaml --checkpoint checkpoints/unet_ce_best.pt
"$PYTHON_BIN" src/evaluate.py --config configs/unet_dice.yaml --checkpoint checkpoints/unet_dice_best.pt
"$PYTHON_BIN" src/evaluate.py --config configs/unet_ce_dice.yaml --checkpoint checkpoints/unet_ce_dice_best.pt
"$PYTHON_BIN" scripts/make_report.py --results runs --output reports/experiment_report.pdf
