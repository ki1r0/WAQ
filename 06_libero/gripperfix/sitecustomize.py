"""Track A registrations: gripper-fixed data + corrected normalization,
features extracted by the RETRAINED VA2 (step ##).
"""
from copy import deepcopy
import json
from pathlib import Path

from wan_va.configs import VA_CONFIGS

STATS_PATH = Path("/path/to/storage/datasets/libero_10_original_500_gripperfix_20260909/action_q01_q99.json")
stats = json.loads(STATS_PATH.read_text())
if stats["episode_count"] != 500 or stats["action_dim"] != 7:
    raise RuntimeError(f"invalid LIBERO-10 normalization contract: {stats}")
if stats["q01"][6] != -1.0 or stats["q99"][6] != 1.0:
    raise RuntimeError(
        "gripper normalization must be the identity (-1=open,+1=close); "
        f"got q01={stats['q01'][6]} q99={stats['q99'][6]} -- wrong stats file"
    )

base = deepcopy(VA_CONFIGS["libero_plus_critic_cache_lingbot_va"])
base.empty_emb_path = "/path/to/assets/empty_emb.pt"
base.dataset_format = "lerobot_v21"
base.env_type = "libero_plus_panda"
base.height = 128
base.width = 128
base.source_fps = 20
base.target_fps = 10
base.temporal_downsample_factor = 2
base.action_fps = 10
base.action_downsample_policy = "stride_2"
base.action_downsample_factor = 2
base.action_downsample_reduction = "stride"
base.frame_chunk_size = 2
base.action_dim = 30
base.action_per_frame = 4
base.used_action_channel_ids = list(range(7))
base.inverse_used_action_channel_ids = [i if i < 7 else 7 for i in range(30)]
base.action_norm_method = "quantiles"
base.norm_stat = {"q01": stats["q01"] + [0.0] * 23, "q99": stats["q99"] + [0.0] * 23}
base.jepa_loss_enabled = True
base.dataset_init_workers = 0
base.load_worker = 0
base.preflight_required = []

open_cfg = deepcopy(base)
open_cfg.__name__ = "Config: VA2-GF official LIBERO-10 500 (gripper-fixed)"
open_cfg.dataset_path = "/path/to/storage/datasets/libero_10_original_500_gripperfix_20260909/lerobot_v21_10hz_vae128"
open_cfg.wan22_pretrained_model_name_or_path = "/path/to/storage/experiments/robotwin_clean_sft_20260903/export/from_va2_gripperfix_20260909_step25000_diffusers"
open_cfg.obs_cam_keys = ["observation.images.image", "observation.images.wrist_image"]
open_cfg.latent_only_video_keys = [
    "observation.images.front", "observation.images.wrist",
    "observation.images.image", "observation.images.wrist_image",
]
VA_CONFIGS["libero_gf_va2_open_500_cache_20260909"] = open_cfg

cosmos_cfg = deepcopy(base)
cosmos_cfg.__name__ = "Config: VA2-GF Cosmos-Policy LIBERO-10 500 (gripper-fixed norm)"
cosmos_cfg.dataset_path = "/path/to/storage/datasets/LIBERO-Cosmos-Policy-lingbotva-20260906-all2000"
cosmos_cfg.wan22_pretrained_model_name_or_path = "/path/to/storage/experiments/robotwin_clean_sft_20260903/export/from_va2_gripperfix_20260909_step25000_diffusers"
cosmos_cfg.obs_cam_keys = ["observation.images.front", "observation.images.wrist"]
cosmos_cfg.latent_only_video_keys = [
    "observation.images.front", "observation.images.wrist",
    "observation.images.image", "observation.images.wrist_image",
]
cosmos_cfg.episode_include_only = list(range(500))
VA_CONFIGS["libero_gf_va2_cosmos_500_cache_20260909"] = cosmos_cfg
