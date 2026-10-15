#!/usr/bin/env python3
"""Create a private inference-only config view for Critic2 checkpoint v1."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: sanitize.py SOURCE_CONFIG DEST_CONFIG RECORD_JSON")
    source, destination, record_path = map(Path, sys.argv[1:])
    source_data = json.loads(source.read_text())
    sys.path.insert(0, "/home/zengqixin/wy/lingbot-va-original-rl-adaptive-cpb")
    from wan_va.rl.config import CriticTrainingConfig

    known = {field.name for field in dataclasses.fields(CriticTrainingConfig)}
    sanitized = {key: value for key, value in source_data.items() if key in known}
    unknown = sorted(set(source_data) - known)
    required = {"base_config_name", "output_dir", "infer_latent_chunk_size"}
    if not required <= sanitized.keys():
        raise ValueError(f"missing required keys after sanitization: {sorted(required - sanitized.keys())}")
    if sanitized.get("critic_type") != "twin_language_query_transformer_v1":
        raise ValueError(f"unexpected critic_type: {sanitized.get('critic_type')!r}")
    if sanitized.get("feature_layers") != [0] or sanitized.get("critic_feature_layers") != [0]:
        raise ValueError("Critic2 v1 layer contract must remain feature_layers=[0], critic_feature_layers=[0]")
    if sanitized.get("infer_latent_chunk_size") != 2:
        raise ValueError("Critic2 v1 chunk contract must remain infer_latent_chunk_size=2")
    destination.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n")
    loaded = CriticTrainingConfig.from_json(destination)
    record = {
        "source_config": str(source),
        "destination_config": str(destination),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "destination_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "source_key_count": len(source_data),
        "destination_key_count": len(sanitized),
        "dropped_unknown_keys": unknown,
        "known_dataclass_key_count": len(known),
        "critic_type": loaded.critic_type,
        "feature_layers": list(loaded.feature_layers),
        "critic_feature_layers": list(loaded.critic_feature_layers),
        "infer_latent_chunk_size": loaded.infer_latent_chunk_size,
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
