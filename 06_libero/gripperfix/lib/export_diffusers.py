#!/usr/bin/env python3
"""Assemble a diffusers export from a VA2 SFT checkpoint.

Two checkpoint formats exist in this project and both are handled here:

* ``clean-resume-v2`` (what the 2026-09-09 gripper-fix run writes): rank 0
  already saved the consolidated bf16 weights under
  ``<ckpt>/transformer/diffusion_pytorch_model.safetensors``.  Nothing needs
  reconstructing -- the transformer is copied and the remaining components are
  symlinked from the reference export.
* DeepSpeed ZeRO-2 (what the 2026-09-07 run wrote): the fp32 state dict is
  reconstructed from the shards, keys the trainer never touched are backfilled
  from the reference export, and everything is cast to the reference dtype.

Either way the exported key set is verified against the reference export
before anything is written to the output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

COMPONENTS = ("assets", "configuration.json", "README.md", "text_encoder", "tokenizer", "vae")
WEIGHTS = "transformer/diffusion_pytorch_model.safetensors"


def keys_of(path: Path) -> list[str]:
    with safe_open(str(path), framework="pt") as f:
        return list(f.keys())


def link_components(out: Path, reference: Path) -> None:
    for name in COMPONENTS:
        src = reference / name
        if src.exists():
            os.symlink(src, out / name)


def direct_assembly(ckpt: Path, reference: Path, out: Path) -> None:
    """Copy the already-consolidated weights, backfilling untrained keys.

    The trainer wraps only the parameters it optimizes, so the checkpoint is
    missing the frozen ``patch_embedding.*`` tensors that the diffusers layout
    expects; they are taken from the reference export exactly as the ZeRO path
    does.
    """
    src = ckpt / WEIGHTS
    ref = load_file(str(reference / WEIGHTS))
    got = load_file(str(src))
    extra = [k for k in got if k not in ref]
    if extra:
        raise SystemExit(f"unexpected keys not present in reference: {extra[:5]}")
    for k in [k for k in ref if k not in got]:
        got[k] = ref[k].clone()
        print(f"backfilled unused key from reference: {k}")
    out_sd = {}
    for k in ref:
        t = got[k].to(ref[k].dtype).contiguous()
        if t.shape != ref[k].shape:
            raise SystemExit(f"shape mismatch on {k}: {t.shape} vs {ref[k].shape}")
        out_sd[k] = t
    (out / "transformer").mkdir(parents=True)
    print(f"writing {len(out_sd)} tensors ...", flush=True)
    save_file(out_sd, str(out / WEIGHTS), metadata={"format": "pt"})
    shutil.copy2(ckpt / "transformer" / "config.json", out / "transformer" / "config.json")
    link_components(out, reference)
    print(f"done: {out}  ({len(out_sd)} tensors, keys verified against reference)")


def zero_consolidation(ckpt_root: Path, tag: str, reference: Path, out: Path) -> None:
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    print(f"consolidating {tag} ...", flush=True)
    sd = get_fp32_state_dict_from_zero_checkpoint(str(ckpt_root), tag=tag)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    print(f"reconstructed {len(sd)} params", flush=True)
    ref = load_file(str(reference / WEIGHTS))
    extra = [k for k in sd if k not in ref]
    if extra:
        raise SystemExit(f"unexpected keys not present in reference: {extra[:5]}")
    for k in [k for k in ref if k not in sd]:
        sd[k] = ref[k].clone()
        print(f"backfilled unused key from reference: {k}")
    out_sd = {}
    for k in ref:
        t = sd[k].to(ref[k].dtype).contiguous()
        if t.shape != ref[k].shape:
            raise SystemExit(f"shape mismatch on {k}: {t.shape} vs {ref[k].shape}")
        out_sd[k] = t
    (out / "transformer").mkdir(parents=True)
    save_file(out_sd, str(out / WEIGHTS), metadata={"format": "pt"})
    shutil.copy2(reference / "transformer" / "config.json", out / "transformer" / "config.json")
    link_components(out, reference)
    print(f"done: {out}  ({len(out_sd)} tensors, keys verified against reference)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="the checkpoint directory itself, e.g. .../checkpoints/checkpoint_step_25000")
    ap.add_argument("--reference-export", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ckpt, reference, out = args.checkpoint, args.reference_export, args.out
    if not (reference / WEIGHTS).is_file():
        raise SystemExit(f"reference transformer missing: {reference / WEIGHTS}")
    if out.exists():
        raise SystemExit(f"refusing pre-existing output: {out}")
    if not ckpt.is_dir():
        raise SystemExit(f"missing checkpoint: {ckpt}")

    if (ckpt / WEIGHTS).is_file():
        if not (ckpt / "COMPLETE").is_file():
            raise SystemExit(f"checkpoint is not marked COMPLETE: {ckpt}")
        state = json.loads((ckpt / "resume_manifest.json").read_text())
        print(f"clean-resume-v2 checkpoint, step={state.get('step')}, "
              f"world_size={state.get('world_size')}")
        direct_assembly(ckpt, reference, out)
    else:
        zero_consolidation(ckpt.parent, ckpt.name, reference, out)
    print("EXPORT_OK")


if __name__ == "__main__":
    main()
