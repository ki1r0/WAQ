"""Isolated Track-A registrations for the orientation+gripper-fixed VA2 25k run.

The two configs intentionally point only at the prepared target roots and
the verified model-only export.  No old gripperfix-only cache or federated
artifact is referenced here.
"""

from copy import deepcopy

from wan_va.configs import VA_CONFIGS


STAGE_TAG = "orientationfix_gripperfix_frame2_20260916"
TARGET_MODEL = (
    "/path/to/storage/checkpoints/va2_libero10_cosmos_orientationfix_gripperfix_"
    "frame2_repro/export_step25000"
)
TARGET_DATASET = (
    "/path/to/storage/datasets/"
    "libero10_va2_success_mix_orientationfix_gripperfix_frame2_final_20260912"
)

Q01_7 = [
    -0.637499988079071,
    -0.768750011920929,
    -0.7473214268684387,
    -0.1039285734295845,
    -0.15214285254478455,
    -0.2786892834305762,
    -1.0,
]
Q99_7 = [
    0.7633928656578064,
    0.84375,
    0.9375,
    0.14785714447498322,
    0.15964286029338837,
    0.32035714387893677,
    1.0,
]
if Q01_7[6] != -1.0 or Q99_7[6] != 1.0:
    raise RuntimeError("orientationfix stage requires identity gripper normalization")


def _base():
    cfg = deepcopy(VA_CONFIGS["libero_plus_critic_cache_lingbot_va"])
    cfg.empty_emb_path = "/path/to/assets/empty_emb.pt"
    cfg.dataset_format = "lerobot_v21"
    cfg.env_type = "libero_plus_panda"
    cfg.height = 128
    cfg.width = 128
    cfg.source_fps = 20
    cfg.target_fps = 10
    cfg.temporal_downsample_factor = 2
    cfg.action_fps = 10
    cfg.action_downsample_policy = "stride_2"
    cfg.action_downsample_factor = 2
    cfg.action_downsample_reduction = "stride"
    cfg.frame_chunk_size = 2
    cfg.action_dim = 30
    cfg.action_per_frame = 4
    cfg.used_action_channel_ids = list(range(7))
    cfg.inverse_used_action_channel_ids = [i if i < 7 else 7 for i in range(30)]
    cfg.action_norm_method = "quantiles"
    cfg.norm_stat = {
        "q01": Q01_7 + [0.0] * 23,
        "q99": Q99_7 + [0.0] * 23,
    }
    cfg.obs_cam_keys = [
        "observation.images.front",
        "observation.images.wrist",
    ]
    cfg.latent_only_video_keys = list(cfg.obs_cam_keys)
    cfg.jepa_loss_enabled = True
    cfg.dataset_init_workers = 0
    cfg.load_worker = 0
    cfg.preflight_required = []
    cfg.wan22_pretrained_model_name_or_path = TARGET_MODEL
    return cfg


open_cfg = _base()
open_cfg.__name__ = (
    "Config: orientation+gripper-fixed VA2 open500 frame2 cache 20260916"
)
open_cfg.dataset_path = f"{TARGET_DATASET}/open500"
VA_CONFIGS["orientationfix_gripperfix_va2_open500_frame2_cache_20260916"] = open_cfg

cosmos_cfg = _base()
cosmos_cfg.__name__ = (
    "Config: orientation+gripper-fixed VA2 Cosmos success376 frame2 cache 20260916"
)
cosmos_cfg.dataset_path = f"{TARGET_DATASET}/cosmos_success376"
cosmos_cfg.episode_include_only = list(range(376))
VA_CONFIGS[
    "orientationfix_gripperfix_va2_cosmos_success376_frame2_cache_20260916"
] = cosmos_cfg
