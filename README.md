# PyTorch 图像分割模型像素级训练

本项目面向 Stanford Background Dataset，从零实现经典 U-Net 语义分割网络，并对比 Cross-Entropy、Dice Loss、Cross-Entropy + Dice 三种损失配置在验证集和测试集上的表现。

## 技术栈

- Python 3.10+
- PyTorch
- Pillow / NumPy
- Matplotlib
- SwanLab（可选实验记录，本地 CSV 和曲线图始终会保存）

## 项目结构

- `src/`：数据集读取、U-Net、损失函数、指标、训练与评估代码。
- `configs/`：三种损失配置。
- `scripts/`：数据准备、报告生成和 Slurm 训练脚本。
- `reports/`：实验报告与图表输出目录。
- `checkpoints/`：最佳模型权重输出目录。

## 环境准备

Windows：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux / Slurm：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## 数据准备

```bash
.venv/bin/python scripts/prepare_data.py --root data
```

如果网络无法访问，也可以手动下载 `iccv09Data.tar.gz` 放到 `data/` 目录后再次运行上述命令。

## 训练

```bash
.venv/bin/python src/train.py --config configs/unet_ce.yaml
.venv/bin/python src/train.py --config configs/unet_dice.yaml
.venv/bin/python src/train.py --config configs/unet_ce_dice.yaml
```

每次训练会保存最佳验证集 mIoU 权重，并在 `runs/` 下保存 CSV 指标、曲线图和配置快照。

Slurm 集群训练优先使用 GPU `aws` 分区：

```bash
sbatch scripts/train_unet_slurm.sh
```

## 评估与报告

```bash
.venv/bin/python src/evaluate.py --config configs/unet_ce_dice.yaml --checkpoint checkpoints/unet_ce_dice_best.pt
.venv/bin/python scripts/make_report.py --results runs --output reports/experiment_report.pdf
```

报告会汇总三种损失配置的训练曲线、验证指标、测试指标和可视化预测结果。

Slurm 集群后处理优先使用非 `aws-com` 的 CPU 分区：

```bash
sbatch scripts/evaluate_report_cpu_slurm.sh
```
