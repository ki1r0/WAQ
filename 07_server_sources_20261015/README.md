# Server script source copies (2026-10-15)

This directory contains 55 unmodified `.py` and `.sh` files copied from the server paths recorded in the local `服务器路径汇总.md`. The directory structure preserves paths relative to the source roots below:

- `reward_ablation/` → `/mnt/zengqixin/datasets/reward_ablation/`
- `lingbot_va/` → `/home/zengqixin/wy/lingbot-va-original-rl/`
- `robocoin_droid_r5/` → `/mnt/zengqixin/experiments/robocoin_droid_critic_controls_20260826/revised_r3_20260918_4node_bs10_r5/`
- `libero/gripperfix/` → `/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/`
- `libero/orientationfix_frame4/` → `/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/libero_open500_orientationfix_va2_critic2_official_20260911/`
- `libero/va2_rebuild/` → `/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/va2_rebuild_20260907/`

Included sources are the confirmed September four-task critic-ablation launchers and trainer, checkpoint-sweep scripts, RoboCOIN/DROID R5 training scripts, the RobotWin clean-to-random evaluator, scripts from the confirmed LIBERO gripperfix and orientationfix experiment directories, and the explicitly identified VA2 cache-step launcher.

Only scripts were copied. Configurations, runbooks, logs, datasets, model weights, checkpoints, caches, and results were excluded. Rows whose source paths remain unconfirmed or ambiguous in the path summary (including the August single-input ablation and VA clean-SFT server source) were not guessed or added. Existing curated files in `01_critic_ablation/` through `06_libero/` were left untouched.
