<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-13 -->
<!-- file-state: revision=5 | updated-at=2026-05-13T13:59:43+08:00 | writer-id=Codex | base-workspace-revision=8 -->

<!-- section: current_state -->
# 当前状态

- 这个课程项目已经完成首版代码实现，包含 U-Net、Stanford Background Dataset 读取、Dice Loss、训练/评估脚本、Slurm 脚本和 PDF 报告生成脚本。
- 本地已完成依赖安装、单元测试和 1 个 epoch 的 smoke test，训练、验证、评估、checkpoint 保存与可视化链路已跑通。
- 真实数据压缩包 `data/iccv09Data.tar.gz` 已放入项目并完成解压；当前存在 `data/raw/` 和时间戳解压目录 `data/raw_20260513_135335/`。
- `.gitignore` 已更新为忽略 `data/*.tar.gz`、`data/raw*/` 和 `data/iccv09Data/`，避免把数据集压缩包与解压数据提交进仓库。

<!-- section: active_judgments -->
# 当前判断

- 当前项目主体已经进入“按真实数据完成正式实验产出”阶段。
- 默认训练输入是项目本地 `.venv`，不依赖激活环境；训练代码会从 `data/` 下自动发现可用的 `iccv09Data/images` 与 `labels`。
- 三组实验配置分别对应 `ce`、`dice`、`ce_dice`，最终需要保留各自验证集 mIoU 最优权重。

<!-- section: risks_open_questions -->
# 风险与未决问题

- 正式训练结果、测试结果和最终 PDF 还没有生成。
- 当前存在两个解压目录，训练发现逻辑会扫描 `data/` 子目录并使用第一个含 `images/` 与 `labels/` 的候选目录；如需减少歧义，可由用户确认后手动清理旧解压目录。
- 如果集群环境没有可用 GPU 或网络，Slurm 训练脚本需要改成可用资源参数或采用本机短训方案。

<!-- section: next_step -->
# 下一步

- 运行一次真实数据 smoke test，确认当前解压目录能被 `discover_samples` 正确识别。
- 然后按 `configs/unet_ce.yaml`、`configs/unet_dice.yaml`、`configs/unet_ce_dice.yaml` 顺序启动正式训练。
- 训练完成后运行 `src/evaluate.py` 和 `scripts/make_report.py`，补齐最终权重与 PDF。

<!-- section: recent_pivots -->
# 近期判断反转

- 数据已从“待放入与待解压”推进为“已解压，下一步可做真实数据检查与正式训练”。
