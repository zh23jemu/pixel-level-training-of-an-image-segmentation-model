# PyTorch 图像分割模型像素级训练

本项目基于 Stanford Background Dataset，从零实现经典 U-Net 语义分割模型，并对比三种损失配置：

- Cross-Entropy
- Dice Loss
- Cross-Entropy + Dice Loss

项目包含完整代码、训练配置、普通 Linux 训练命令、Slurm 集群训练脚本、测试评估、预测可视化和 PDF 实验报告生成流程。

## 交付内容

当前项目已产出以下交付物：

- 代码：`src/`、`scripts/`、`configs/`、`tests/`
- 实验报告：`reports/experiment_report.pdf`
- 模型权重：
  - `checkpoints/unet_ce_best.pt`
  - `checkpoints/unet_dice_best.pt`
  - `checkpoints/unet_ce_dice_best.pt`

## 技术栈

- Python 3.10+
- PyTorch
- NumPy
- Pillow
- Matplotlib
- PyYAML
- ReportLab
- tqdm
- SwanLab（可选；即使不用 SwanLab，本地 CSV 指标和曲线图也会保存）

## 项目结构

```text
.
├── configs/
│   ├── unet_ce.yaml
│   ├── unet_dice.yaml
│   └── unet_ce_dice.yaml
├── scripts/
│   ├── prepare_data.py
│   ├── train_unet_slurm.sh
│   ├── evaluate_report_cpu_slurm.sh
│   └── make_report.py
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│   └── visualize.py
├── tests/
├── checkpoints/
├── runs/
└── reports/
```

主要模块说明：

- `src/model.py`：从零实现 U-Net 网络结构。
- `src/dataset.py`：读取 Stanford Background Dataset 图像与 `.regions.txt` 标签，并完成样本划分。
- `src/losses.py`：实现 Cross-Entropy、Dice Loss 和组合损失。
- `src/metrics.py`：计算 Pixel Accuracy、per-class IoU 和 mIoU。
- `src/train.py`：训练入口，保存最佳验证 mIoU 权重、训练 CSV、曲线图和配置快照。
- `src/evaluate.py`：测试集评估入口，保存测试指标和预测可视化。
- `scripts/make_report.py`：根据训练与测试结果生成 PDF 实验报告。
- `scripts/train_unet_slurm.sh`：Slurm GPU 训练脚本。
- `scripts/evaluate_report_cpu_slurm.sh`：Slurm CPU 后处理脚本，用于测试评估和报告生成。

## 环境准备

推荐始终使用项目本地虚拟环境 `.venv`，不要依赖系统 Python 或 shell 激活状态。

### 普通 Linux

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

如果系统没有 Python 3.11，也可以使用 Python 3.10：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Slurm 集群

在集群登录节点进入项目根目录后，先创建并安装本地 `.venv`：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

如集群没有 Python 3.11，可改用 Python 3.10。

## 数据准备

项目使用 Stanford Background Dataset。默认数据根目录为 `data/`。

### 自动下载并解压

```bash
.venv/bin/python scripts/prepare_data.py --root data
```

### 已有压缩包时跳过下载

如果已经有 `iccv09Data.tar.gz`，将其放到 `data/` 后运行：

```bash
.venv/bin/python scripts/prepare_data.py --root data --skip-download
```

准备完成后，训练代码会自动发现可用的图像目录和标签目录。

## 实验配置

三组实验配置位于 `configs/`：

| 配置文件 | 实验名 | 损失函数 | 输出目录 | 最佳权重 |
|---|---|---|---|---|
| `configs/unet_ce.yaml` | `unet_ce` | Cross-Entropy | `runs/unet_ce/` | `checkpoints/unet_ce_best.pt` |
| `configs/unet_dice.yaml` | `unet_dice` | Dice Loss | `runs/unet_dice/` | `checkpoints/unet_dice_best.pt` |
| `configs/unet_ce_dice.yaml` | `unet_ce_dice` | Cross-Entropy + Dice | `runs/unet_ce_dice/` | `checkpoints/unet_ce_dice_best.pt` |

默认训练参数：

- 输入尺寸：`240 x 320`
- 类别数：`8`
- batch size：`8`
- epoch：`40`
- 学习率：`0.001`
- 优化器：Adam
- 设备：`auto`，优先使用 CUDA
- 数据划分：训练集 70%，验证集 15%，测试集 15%

## 普通 Linux 训练

在项目根目录依次运行三组配置：

```bash
.venv/bin/python src/train.py --config configs/unet_ce.yaml
.venv/bin/python src/train.py --config configs/unet_dice.yaml
.venv/bin/python src/train.py --config configs/unet_ce_dice.yaml
```

如果只想做快速 smoke test，可以临时覆盖 epoch 数：

```bash
.venv/bin/python src/train.py --config configs/unet_ce.yaml --epochs 1 --no-swanlab
```

训练完成后会生成：

```text
checkpoints/*_best.pt
runs/*/metrics.csv
runs/*/summary.json
runs/*/loss.png
runs/*/accuracy.png
runs/*/miou.png
runs/*/config.yaml
runs/*/resolved_config.json
```

## Slurm 训练

GPU 训练脚本为：

```bash
scripts/train_unet_slurm.sh
```

当前脚本默认资源配置：

- 分区：`aws`
- account：`gpo-ifv7xx`
- GPU：`1`
- CPU：`4`
- 内存：`16G`
- 时间：`1-00:00:00`

提交训练：

```bash
sbatch scripts/train_unet_slurm.sh
```

查询状态：

```bash
squeue -j <job_id>
```

训练日志会写入：

```text
logs/slurm/unet_sbd-<job_id>.out
logs/slurm/unet_sbd-<job_id>.err
```

脚本会按顺序训练：

1. `configs/unet_ce.yaml`
2. `configs/unet_dice.yaml`
3. `configs/unet_ce_dice.yaml`

## 普通 Linux 测试评估与报告生成

三组训练完成后，分别评估测试集：

```bash
.venv/bin/python src/evaluate.py --config configs/unet_ce.yaml --checkpoint checkpoints/unet_ce_best.pt
.venv/bin/python src/evaluate.py --config configs/unet_dice.yaml --checkpoint checkpoints/unet_dice_best.pt
.venv/bin/python src/evaluate.py --config configs/unet_ce_dice.yaml --checkpoint checkpoints/unet_ce_dice_best.pt
```

测试结果默认写入对应 run 目录下的 `test/` 子目录：

```text
runs/unet_ce/test/metrics.json
runs/unet_ce/test/prediction_00.png

runs/unet_dice/test/metrics.json
runs/unet_dice/test/prediction_00.png

runs/unet_ce_dice/test/metrics.json
runs/unet_ce_dice/test/prediction_00.png
```

默认每组最多保存 8 张预测可视化。如需修改数量：

```bash
.venv/bin/python src/evaluate.py \
  --config configs/unet_ce.yaml \
  --checkpoint checkpoints/unet_ce_best.pt \
  --max-visualizations 12
```

生成 PDF 实验报告：

```bash
.venv/bin/python scripts/make_report.py --results runs --output reports/experiment_report.pdf
```

报告内容包括：

- 实验设置概述
- 三组损失配置的验证指标和测试指标
- loss、accuracy、mIoU 曲线
- 测试集预测可视化
- 结论说明

## Slurm 测试评估与报告生成

训练完成后提交 CPU 后处理脚本：

```bash
sbatch scripts/evaluate_report_cpu_slurm.sh
```

当前脚本默认资源配置：

- 分区：`defq`
- account：`gpo-ifv7xx`
- CPU：`4`
- 内存：`16G`
- 时间：`02:00:00`

查询状态：

```bash
squeue -j <job_id>
```

日志会写入：

```text
logs/slurm/unet_sbd_eval-<job_id>.out
logs/slurm/unet_sbd_eval-<job_id>.err
```

成功后应生成：

```text
runs/unet_ce/test/metrics.json
runs/unet_ce/test/prediction_*.png
runs/unet_dice/test/metrics.json
runs/unet_dice/test/prediction_*.png
runs/unet_ce_dice/test/metrics.json
runs/unet_ce_dice/test/prediction_*.png
reports/experiment_report.pdf
```

## 当前实验结果

本项目当前测试集结果如下：

| 实验 | Test Pixel Accuracy | Test mIoU |
|---|---:|---:|
| CE | 0.6545 | 0.4187 |
| Dice | 0.6174 | 0.4022 |
| CE + Dice | 0.6403 | 0.4150 |

测试集上 CE 的 mIoU 最高，CE + Dice 非常接近，Dice 单独训练最低。验证集上 CE + Dice 略优于 CE，因此最终报告中保留了验证集与测试集表现差异。

## 测试

运行单元测试：

```bash
.venv/bin/python -m pytest
```

当前测试覆盖核心数据流、损失函数、指标计算和模型前向传播。

## 常见问题

### PDF 中文显示为方块

`scripts/make_report.py` 已包含中文字体注册逻辑：

- Windows 优先使用微软雅黑或宋体。
- Linux 优先查找 Noto CJK 或 AR PL 字体。
- 如果系统字体不存在，则使用 ReportLab 内置 `STSong-Light` 作为兜底。

如果仍然显示异常，可安装 Noto CJK 字体后重新生成报告。

### Slurm 提示 account 或分区不可用

检查脚本中的：

```bash
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
```

以及后处理脚本中的：

```bash
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
```

如集群策略变化，需要根据实际账号权限调整。

### 训练日志里出现 NFS 临时文件清理报错

如果 checkpoint、训练曲线、测试指标和报告都已生成，这类报错通常是集群文件系统清理临时文件时的环境噪声。建议优先检查最终产物是否完整。

## 最终提交检查清单

提交前建议确认：

- `src/`、`scripts/`、`configs/` 代码完整。
- 三份最佳模型权重位于 `checkpoints/`。
- 三组训练结果位于 `runs/`。
- 三组测试评估和预测图位于 `runs/*/test/`。
- PDF 报告位于 `reports/experiment_report.pdf`。
- 报告中文字、表格、曲线和预测图显示正常。
