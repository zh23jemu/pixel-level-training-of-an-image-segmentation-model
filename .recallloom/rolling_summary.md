<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [RecallLoom] | 2026-05-14 -->
<!-- file-state: revision=14 | updated-at=2026-05-14T07:04:36+08:00 | writer-id=RecallLoom | base-workspace-revision=23 -->

<!-- section: current_state -->
# 当前状态

- 这个课程项目已经完成首版代码实现，包含 U-Net、Stanford Background Dataset 读取、Dice Loss、训练评估脚本、Slurm 脚本和 PDF 报告生成脚本。
- 本地已完成依赖安装、单元测试和 1 个 epoch 的 smoke test，训练、验证、评估、checkpoint 保存与可视化链路已跑通。
- 真实数据压缩包已放入项目并完成解压；真实数据读取检查通过，样本数为 715。
- GPU 训练作业 32953705 已在服务器端跑完，训练输出已同步到本地并初步验证完整。
- CPU 后处理作业 33019435 已完成，三组测试指标、每组 8 张预测可视化和 PDF 实验报告已同步到本地。
- 已修复 PDF 报告中文乱码问题：`scripts/make_report.py` 现在会注册中文字体并为表格显式设置字体，`reports/experiment_report.pdf` 已重新生成。

<!-- section: active_judgments -->
# 当前判断

- 当前项目已完成训练、测试评估、预测可视化和报告生成的主要交付闭环。
- 测试 mIoU 排名为：CE 0.4187，CE 加 Dice 0.4150，Dice 0.4022；测试集上 CE 表现最好。
- 验证集上 CE 加 Dice 略优于 CE，但测试集上 CE 略优；最终报告中应如实呈现验证和测试差异。
- 下一步不需要继续训练，重点转为人工检查新版 PDF 的中文、排版、预测图视觉质量和提交范围。

<!-- section: risks_open_questions -->
# 风险与未决问题

- 训练日志中存在 CUDA driver warning 和 NFS 临时文件清理报错，但训练与后处理产物已生成，当前更像集群环境噪声。
- 新版 PDF 已重新生成，但还需要用户人工确认中文、表格、曲线和预测图是否完全符合课程提交要求。
- 还未确认是否需要 Git 提交、打包或最终提交说明。

<!-- section: next_step -->
# 下一步

- 人工打开 `reports/experiment_report.pdf` 检查中文是否正常、表格是否清晰、曲线和预测图是否正常。
- 如果服务器也需要保留新版报告，拉取脚本修复后在服务器重新运行报告生成或 CPU 后处理脚本。
- 确认最终提交范围；如用户要求提交 git，则检查 `.gitignore` 后按中文 Conventional Commits 提交。

<!-- section: recent_pivots -->
# 近期判断反转

- GPU 训练脚本已经提交并在服务器端跑完，训练产物已初步验证完整。
- CPU 后处理脚本已经提交并完成，测试指标、预测可视化和报告已生成。
- 原 PDF 报告存在中文乱码，已修复字体注册与表格字体设置并重新生成报告。
