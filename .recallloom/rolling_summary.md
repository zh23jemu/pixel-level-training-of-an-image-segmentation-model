<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-13 -->
<!-- file-state: revision=8 | updated-at=2026-05-13T15:39:40+08:00 | writer-id=Codex | base-workspace-revision=11 -->

<!-- section: current_state -->
# 当前状态

- 这个课程项目已经完成首版代码实现，包含 U-Net、Stanford Background Dataset 读取、Dice Loss、训练评估脚本、Slurm 脚本和 PDF 报告生成脚本。
- 本地已完成依赖安装、单元测试和 1 个 epoch 的 smoke test，训练、验证、评估、checkpoint 保存与可视化链路已跑通。
- 真实数据压缩包已放入项目并完成解压；真实数据读取检查通过，样本数为 715。
- Slurm 脚本已按用户集群环境适配：GPU 训练使用 aws 分区，CPU 后处理使用 defq 分区，并显式指定 account 为 gpo-ifv7xx。

<!-- section: active_judgments -->
# 当前判断

- 当前项目已进入正式实验阶段，下一步可以在集群上重新提交训练作业。
- 默认训练输入是项目本地虚拟环境，训练代码会自动发现可用的图像目录与标签目录。
- 三组实验配置分别对应 CE、Dice、CE 加 Dice，最终需要保留各自验证集 mIoU 最优权重。
- 集群 Python 虚拟环境建议使用 Python 3.11；若无 3.11，则退到 Python 3.10。

<!-- section: risks_open_questions -->
# 风险与未决问题

- 正式训练结果、测试结果和最终 PDF 还没有生成。
- 如果 account 和分区组合仍被拒绝，需要在集群上用完整 sacctmgr 输出核对该 account 是否允许 aws 分区。

<!-- section: next_step -->
# 下一步

- 重新提交 GPU 训练脚本。
- 训练完成后提交 CPU 后处理脚本生成评估结果和报告。

<!-- section: recent_pivots -->
# 近期判断反转

- 集群拒绝默认 account 后，已参考另一个项目的 Slurm 脚本，显式补入 account。
