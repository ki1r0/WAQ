# WAQ / LingBot-VA

## 目录

- `01_critic_ablation/`：8 月单输入 reward-only（Pi0.6 time-cost、success+goal-progress）与 9 月四任务 reward-term 消融；含训练入口、配置、缓存索引和评测适配代码。
- `02_checkpoint_sweep/`：RobotWin checkpoint TD sweep、old-language QTran-L 指标评测，以及 Pi0.6 reward 的离线 TD/Q-AUC 分析。
- `03_robocoin_droid_pretraining/`：RoboCOIN/DROID 联合预训练。
- `04_va_clean_sft/`：VA clean SFT。
- `05_robotwin_clean_to_random_eval/`：Base no-Q / Joint guided 评测。
- `06_libero/`：gripperfix（含 Critic1 baseline、VA2/Critic2 与 no-Q/guided eval）、orientationfix frame4/action4（每 chunk 16 actions）、frame2/10Hz 前处理与训练、后续 Critic2。
- `07_server_sources_20261015/`：按服务器路径汇总中已确认的源路径，从服务器逐字节归档的 55 个 `.py`/`.sh` 脚本；按实验类别保留相对层级，映射见目录内说明，未覆盖前述整理版。

只收代码与配置模板；未收真机材料、CRL/SeqFT、数据/权重/缓存、日志或近期结果汇总。原有整理目录中的配置路径已替换为 `/path/to/...`，运行前需按目标环境补齐外部依赖与路径；`07_server_sources_20261015/` 则保留服务器脚本原样及来源路径映射。8 月训练入口引用的原始冻结源码快照未找到，因此该部分是脚本与配置归档，不保证脱离原运行环境可直接复现。
