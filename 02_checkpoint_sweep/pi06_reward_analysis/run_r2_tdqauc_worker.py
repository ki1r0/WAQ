#!/usr/bin/env python3
"""One-GPU sequential worker for the fixed R2 offline TD/AUC checkpoint sweep."""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
root=Path('/path/to/mnt/analysis/robocoin_r2_pi06exact_hangingmug_td_qauc_sweep_20260828')
adapter_root=Path('/path/to/mnt/checkpoints/eval_adapters/ROBOCOIN-DROID-R2-PI06EXACT-HANGINGMUG-HARD-ONLINEQ-20260827-R1')
scorer=root/'code/r2_pi06exact_offline_td_qauc.py'
replay=Path('/path/to/mnt2/analysis/robotwin_sft_td_checkpoint_sweep_20260812/replays/hanging_mug')
clean_cache=Path('/path/to/mnt2/analysis/robotwin_sft_td_checkpoint_sweep_20260812/clean_cache/hanging_mug')
clean_root=Path('/path/to/home/wy/giga/dataset/lingbot-robotwin-clean-and-aug-lerobot/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50/hanging_mug-demo_clean_collect_200-50')
reward=Path('/path/to/mnt2/analysis/robotwin_sft_td_checkpoint_sweep_20260812/frozen/reward_metadata.pt')
catalog='a722b05487e244293aa10fca6a81b8a3eed2b1a30a35b1ec4dd6e74dde2bd977'
for raw in sys.argv[1:]:
    step=int(raw); out=root/'scores'/f'checkpoint_step_{step}'
    if (out/'td_error_curves.json').is_file():
        print(f'SKIP {step} complete',flush=True); continue
    if out.exists():
        raise RuntimeError(f'partial output exists: {out}')
    cmd=[sys.executable,str(scorer),'score','--task','hanging_mug','--checkpoint',str(adapter_root/f'checkpoint_step_{step}'),'--sft-replay',str(replay),'--clean-root',str(clean_root),'--clean-cache-dir',str(clean_cache)]
    for episode in range(10): cmd += ['--clean-episode',str(episode)]
    cmd += ['--reward-metadata',str(reward),'--expected-catalog-sha256',catalog,'--output',str(out)]
    print('START',step,flush=True)
    subprocess.run(cmd,check=True)
    print('DONE',step,flush=True)
