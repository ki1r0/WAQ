from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VJEPA_ROOT = REPO_ROOT.parent / "vjepa2"
for path in (str(VJEPA_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def build_jepa_encoder(
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    from app.vjepa_2_1.models.vision_transformer import vit_gigantic_xformers

    encoder = vit_gigantic_xformers(
        patch_size=16,
        img_size=(256, 256),
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _clean_state_dict(checkpoint["target_encoder"])
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing:
        print(
            "[JEPA] Warning - "
            f"{len(missing)} missing keys (first 3: {missing[:3]})",
            flush=True,
        )
    if unexpected:
        print(
            "[JEPA] Warning - "
            f"{len(unexpected)} unexpected keys (first 3: {unexpected[:3]})",
            flush=True,
        )

    encoder = encoder.to(device=device, dtype=torch.float32)
    encoder.eval()
    return encoder
