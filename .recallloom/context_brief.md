<!-- recallloom:file=context_brief version=1.0 lang=zh-CN -->
<!-- file-state: revision=2 | updated-at=2026-05-13T13:43:53+08:00 | writer-id=Codex | base-workspace-revision=2 -->

<!-- section: mission -->
# 项目使命

- 本项目用于完成“图像分割模型的像素级训练”课程任务：基于 Stanford Background Dataset，用 PyTorch 从零实现经典 U-Net，并对比 Cross-Entropy、Dice Loss、Cross-Entropy + Dice 三种损失配置。
- 最终交付目标包括：完整代码、PDF 实验报告、三种损失配置训练得到的最佳模型权重。

<!-- section: audience_stakeholders -->
# 受众与相关方

- 主要受众是课程作业评阅者和后续接手实现/训练的工程代理。
- 用户偏好简体中文说明，代码新增复杂逻辑应带较详细中文注释。

<!-- section: current_phase -->
# 当前阶段

- 项目已完成首版代码实现和本地 smoke test，下一阶段是准备真实数据集并在 Slurm 或本机完成正式训练、评估与报告生成。

<!-- section: scope -->
# 范围

- 范围内：PyTorch 原生 U-Net、Stanford Background Dataset 数据读取、三种损失实验、mIoU/Accuracy 指标、预测可视化、PDF 报告、Slurm 脚本。
- 范围外：使用任何预训练权重、引入现成分割框架替代手写 U-Net、删除用户文件或执行破坏性系统命令。

<!-- section: source_of_truth -->
# 事实来源

- 仓库代码与配置是实现事实来源。
- 用户提供的作业截图和计划文本是需求事实来源。
- 数据集压缩包 `iccv09Data.tar.gz` 是训练数据事实来源。

<!-- section: core_workflow -->
# 核心工作流

- 使用项目本地 `.venv` 执行 Python 命令，不依赖激活环境。
- 数据准备：将 `iccv09Data.tar.gz` 放入 `data/` 后运行 `scripts/prepare_data.py`。
- 训练：分别运行 `configs/unet_ce.yaml`、`configs/unet_dice.yaml`、`configs/unet_ce_dice.yaml`。
- 评估与报告：运行 `src/evaluate.py` 保存测试指标和可视化，再运行 `scripts/make_report.py` 生成 PDF。

<!-- section: boundaries -->
# 边界与约束

- 遵守项目 AGENTS 规则：尽量少用 shell，不直接删除文件，编辑前先读取文件，保持最小修改。
- Python 相关命令必须使用项目本地 `.venv`。
- `.gitignore` 应随项目真实内容维护，避免过度泛化。
