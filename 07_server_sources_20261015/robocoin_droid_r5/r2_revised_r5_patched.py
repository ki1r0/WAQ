"""Window-sequential joint RoboCOIN+DROID pretraining.

For each VA window, every legal TD transition is consumed in chronological
order.  The earlier transitions update the critic head from a read-only
shared-backbone feature pass.  The window's final legal transition is then
combined with that window's L_dyn and L_in in one joint backward pass.  Thus
VA losses are computed once per window, while every critic transition receives
one critic update and only the time-last transition also updates the shared
LingBot backbone.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data._utils.collate import default_collate

# See control_train.py: the checkpoint-only gradient-alignment probe needs
# retained access to a compiled graph, so avoid AOTAutograd buffer donation.
try:
    torch._functorch.config.donated_buffer = False
except AttributeError:
    pass

from wan_va.configs import VA_CONFIGS
from wan_va.distributed.util import init_distributed
from wan_va.rl.config import CriticTrainingConfig
from wan_va.rl.algorithms import iql_losses
from wan_va.rl.train_critic import _flatten_action_context, _flatten_video_context
from wan_va.rl.train_joint import JointTrainer
from wan_va.rl.train_joint_alltransitions_pluscritic_sharedbackbone import (
    build_window_local_records,
    make_schedule,
    _take_frames,
)

def metadata_from_pairs(pairs, device):
    """Create the established in-window IQL metadata from (owner, record)."""
    if not pairs:
        return None
    return {
        "owner": torch.tensor([owner for owner, _ in pairs], device=device, dtype=torch.long),
        "frame_start": torch.tensor([record.frame_start for _, record in pairs], device=device, dtype=torch.long),
        "valid_frames": torch.tensor([record.valid_frames for _, record in pairs], device=device, dtype=torch.long),
        "reward": torch.tensor(
            [record.reward for _, record in pairs], device=device, dtype=torch.float32
        ),
        "discount": torch.tensor([record.discount for _, record in pairs], device=device, dtype=torch.float32),
        "done": torch.tensor([record.done for _, record in pairs], device=device, dtype=torch.bool),
    }


def split_window_transitions(owned_records, all_records, entries):
    """Return chronological critic-only rounds and one final joint record/window.

    For a local VA batch, transition i from every window is placed in the same
    critic mini-batch.  This keeps the configured batch size while preserving
    chronology within each individual window.  The final transition of each
    window is reserved for the single L_dyn + L_in + L_c joint update.
    """
    rounds = []
    final_pairs = []
    total = 0
    for owner, (dataset_index, is_real) in enumerate(entries):
        if not is_real:
            continue
        record_indices = sorted(
            owned_records.get(int(dataset_index), []),
            key=lambda index: int(all_records[index].frame_start),
        )
        if not record_indices:
            continue
        total += len(record_indices)
        final_pairs.append((owner, all_records[record_indices[-1]]))
        for round_index, record_index in enumerate(record_indices[:-1]):
            while len(rounds) <= round_index:
                rounds.append([])
            rounds[round_index].append((owner, all_records[record_index]))
    return rounds, final_pairs, total


def global_max_rounds(local_rounds, device):
    rounds = torch.tensor([local_rounds], device=device, dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(rounds, op=dist.ReduceOp.MAX)
    return int(rounds.item())


def sync_critic_gradients(trainer, world_size):
    """Synchronize all trainable critic gradients with one collective.

    The critic lives outside the DeepSpeed engine, so its gradients need an
    explicit cross-rank reduction.  Flattening them gives every rank exactly
    one critic-gradient collective per update, avoiding a long, fragile
    per-parameter collective sequence when some local critic batches are
    empty.
    """
    parameters = [
        parameter
        for parameter in trainer.critic_bundle.parameters()
        if parameter.requires_grad
    ]
    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradients.append(parameter.grad.reshape(-1))
    if world_size > 1:
        flat = torch.cat(gradients)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for parameter in parameters:
            next_offset = offset + parameter.numel()
            parameter.grad.copy_(flat[offset:next_offset].view_as(parameter))
            offset = next_offset
    return parameters


def broadcast_critic_bundle_and_check(trainer, world_size, device):
    """Make external Q/V/EMA state identical; DeepSpeed does not own it."""
    if not dist.is_initialized() or world_size == 1:
        return {"critic_fingerprint_consistent": True}
    state = trainer.critic_bundle.state_dict()
    for tensor in state.values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=0)
    fp = torch.zeros(2, dtype=torch.float64, device=device)
    for tensor in state.values():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            flat = tensor.detach().reshape(-1).double()
            fp[0] += flat.sum(); fp[1] += flat.square().sum()
    gathered = [torch.empty_like(fp) for _ in range(world_size)]
    dist.all_gather(gathered, fp)
    all_fp = torch.stack(gathered)
    if not torch.allclose(all_fp, all_fp[:1].expand_as(all_fp), rtol=0.0, atol=0.0):
        raise RuntimeError("critic Q/V/EMA state mismatch after rank-0 broadcast")
    return {"critic_fingerprint_consistent": True, "critic_fingerprint": [round(float(x), 6) for x in fp.cpu()]}


def transition_scale(local_count, device, world_size):
    local = torch.tensor(float(local_count), device=device)
    global_count = local.clone()
    if dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if global_count.item() == 0:
        return local * 0.0, 0
    return local * float(world_size) / global_count, int(global_count.item())


def all_transition_stats(local_vectors, device):
    """Global means and quantiles over every TD transition in one VA step."""
    local = {key: torch.cat(values).detach().float().cpu().tolist() for key, values in local_vectors.items() if values}
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size(); dist.all_gather_object(gathered, local)
    else: gathered = [local]
    output = {}
    for key in ("q", "value", "target", "td_abs"):
        data = [value for entry in gathered for value in entry.get(key, [])]
        if not data:
            output[key] = {"mean": None, "p10": None, "p50": None, "p90": None}
            continue
        t = torch.tensor(data, dtype=torch.float32)
        output[key] = {"mean":round(float(t.mean()),6),"p10":round(float(torch.quantile(t,.1)),6),"p50":round(float(torch.quantile(t,.5)),6),"p90":round(float(torch.quantile(t,.9)),6)}
    return output


def all_transition_loss_averages(local_records, device):
    """Per-outer-step, transition-count weighted loss means across all ranks.

    R2 performs several chronological critic optimizer updates for each VA
    window.  The historical ``*_last_transition_rank0`` fields describe only
    the final joint update.  This helper reports the sample-weighted mean of
    every critic update that occurred in the same outer VA step.
    """
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    for count, total_loss, q_loss, value_loss in local_records:
        totals += torch.tensor(
            [count, count * total_loss, count * q_loss, count * value_loss],
            dtype=torch.float64, device=device,
        )
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = totals[0].clamp_min(1.0)
    return {
        "transition_count_global": int(totals[0].item()),
        "critic_loss_mean": round(float((totals[1] / count).item()), 6),
        "q_loss_mean": round(float((totals[2] / count).item()), 6),
        "value_loss_mean": round(float((totals[3] / count).item()), 6),
    }


def fixed_window_validation(trainer, window_records, records, world_size, batch_size, seed):
    # Use one latent-length group.  Tail windows are shorter and cannot be
    # default-collated with full 31-latent windows.
    groups=defaultdict(list)
    base=trainer.critic_dataset.base_dataset
    for index in window_records:
        groups[int(base.get_rl_segment_metadata(index)["latent_frame_count"])].append(index)
    _,ids=max(groups.items(),key=lambda pair:(len(pair[1]),pair[0]))
    rng=random.Random(seed+991); rng.shuffle(ids)
    ids=ids[:min(len(ids),world_size*batch_size)]
    rank=int(os.getenv("RANK","0")); local_ids=ids[rank*batch_size:(rank+1)*batch_size]
    if not local_ids: return None
    batch=default_collate([trainer.critic_dataset.base_dataset[i] for i in local_ids]); batch={k:v.to(trainer.device,non_blocking=True) for k,v in batch.items()}
    pairs=[(owner,records[record_idx]) for owner,idx in enumerate(local_ids) for record_idx in window_records.get(int(idx),[])]
    metadata=metadata_from_pairs(pairs,trainer.device)
    with torch.no_grad():
        inputs=trainer._prepare_va_input(batch); output=trainer.transformer(inputs,train_mode=True,return_features=True,critic_feature_layer=trainer.critic_config.feature_layer)
        loss,metrics=diagnostic_critic_loss(trainer,batch,output[3],metadata)
    local=float(loss)*len(pairs); count=torch.tensor(float(len(pairs)),device=trainer.device); total=torch.tensor(local,device=trainer.device)
    if dist.is_initialized(): dist.all_reduce(total,op=dist.ReduceOp.SUM);dist.all_reduce(count,op=dist.ReduceOp.SUM)
    return {"loss":round(float(total/count.clamp_min(1)),6),"transitions":int(count.item()),"stats":all_transition_stats({k:[metrics[k]] for k in ("q","value","target","td_abs")},trainer.device)}


def shared_grad_alignment(trainer, va_loss, critic_loss):
    # The compiled Wan graph allows one real backward only.  The exact
    # VA/Lc cosine belongs in a checkpoint-side, uncompiled probe.
    return {"shared_grad_alignment": "deferred_to_uncompiled_validation_probe"}


def diagnostic_critic_loss(trainer, batch, features, metadata):
    """One representation definition for every earlier and final TD update."""
    if metadata is None:
        z = features["video_tokens"].sum() * 0.0
        return z, {"q_loss": z.detach(), "value_loss": z.detach(), "q": z.detach().reshape(1), "value": z.detach().reshape(1), "target": z.detach().reshape(1), "td_abs": z.detach().reshape(1)}
    width = int(trainer.critic_config.infer_latent_chunk_size)
    owner, starts, valid = metadata["owner"], metadata["frame_start"], metadata["valid_frames"]
    if (starts < width).any(): raise RuntimeError("critic transition lacks its prior latent state")
    video, current_frames = _take_frames(features["video_tokens"], owner, starts, width)
    action, _ = _take_frames(features["action_tokens"], owner, starts, width)
    previous, previous_frames = _take_frames(features["video_tokens"], owner, starts-width, width)
    offsets = torch.arange(width, device=trainer.device)[None, :]; valid_time = offsets < valid[:, None]
    source_action_mask = batch["actions_mask"].to(trainer.device, non_blocking=True)
    source_latent_mask = batch["latents_mask"].to(trainer.device, non_blocking=True)
    current_action_mask = source_action_mask.permute(0,2,1,3,4)[owner[:,None],current_frames].permute(0,2,1,3,4) & valid_time[:,None,:,None,None]
    current_latent_mask = source_latent_mask[owner[:,None],current_frames] & valid_time
    previous_latent_mask = source_latent_mask[owner[:,None],previous_frames]
    action_tokens, action_mask = _flatten_action_context(action,current_action_mask)
    q1,q2=trainer.critic_bundle.q(action_tokens,action_mask)
    previous_tokens,previous_mask=_flatten_video_context(previous,previous_latent_mask)
    value=trainer.critic_bundle.value(previous_tokens,previous_mask)
    with torch.no_grad():
        current_tokens,current_mask=_flatten_video_context(video,current_latent_mask)
        target_value=trainer.critic_bundle.target_value(current_tokens,current_mask)
    losses=iql_losses(q1=q1,q2=q2,value=value,next_target_value=target_value,reward=metadata["reward"],discount=metadata["discount"],expectile=trainer.critic_config.expectile,value_weight=trainer.critic_config.value_loss_weight,value_mask=torch.ones_like(metadata["done"],dtype=torch.bool),q_loss_type=trainer.critic_config.q_loss_type,q_huber_delta=trainer.critic_config.q_huber_delta)
    target=losses.target.detach(); q=.5*(q1.detach()+q2.detach()); td=.5*((q1.detach()-target).abs()+(q2.detach()-target).abs())
    return losses.total,{"q_loss":losses.q.detach(),"value_loss":losses.value.detach(),"q":q,"value":value.detach(),"target":target,"td_abs":td}


def _gradient_connected_zero_for_critic(trainer, loss):
    """Keep an empty local critic batch in the distributed update contract.

    Earlier critic rounds deliberately share a no-grad transformer feature
    pass.  If a rank owns no transition in such a round, the generic loss
    helper returns a detached numeric zero.  That rank must still execute the
    same gradient reduction as every peer, with zero critic gradients.
    """
    if loss.requires_grad:
        return loss
    zero_terms = [
        parameter.reshape(-1)[0] * 0.0
        for parameter in trainer.critic_bundle.parameters()
        if parameter.requires_grad
    ]
    if not zero_terms:
        raise RuntimeError("critic bundle has no trainable parameters")
    return torch.stack(zero_terms).sum()


def critic_head_step(trainer, batch, features, metadata, world_size):
    """One critic-only update; shared transformer remains read-only."""
    trainer.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss, metrics = diagnostic_critic_loss(trainer, batch, features, metadata)
    critic_loss = _gradient_connected_zero_for_critic(trainer, critic_loss)
    scale, global_count = transition_scale(0 if metadata is None else len(metadata["owner"]), trainer.device, world_size)
    if global_count == 0:
        # Every rank observes the same all-reduced count; do not advance the
        # critic optimizer or its EMA/target state for an empty global batch.
        metrics["critic_grad_norm"] = torch.zeros((), device=trainer.device)
        metrics["global_count"] = 0
        return critic_loss.detach(), metrics
    (critic_loss * scale).backward()
    parameters = sync_critic_gradients(trainer, world_size)
    norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=float(trainer.critic_config.max_grad_norm or float("inf")),
    )
    trainer.critic_optimizer.step()
    trainer.critic_bundle.update_target(trainer.critic_config.target_ema_rate)
    metrics["critic_grad_norm"] = norm.detach()
    metrics["global_count"] = global_count
    return critic_loss.detach(), metrics


def joint_final_step(trainer, batch, metadata, world_size, measure_shared_grad=False):
    """The one per-window update that jointly changes VA and critic/backbone."""
    trainer.critic_optimizer.zero_grad(set_to_none=True)
    inputs = trainer._prepare_va_input(batch)
    output = trainer.transformer(
        inputs,
        train_mode=True,
        return_features=True,
        critic_feature_layer=trainer.critic_config.feature_layer,
    )
    latent_loss, action_loss = trainer._va_loss(inputs, output[:2])
    critic_loss, critic_metrics = diagnostic_critic_loss(trainer, batch, output[3], metadata)
    critic_loss = _gradient_connected_zero_for_critic(trainer, critic_loss)
    scale, global_count = transition_scale(0 if metadata is None else len(metadata["owner"]), trainer.device, world_size)
    if measure_shared_grad:
        critic_metrics.update(shared_grad_alignment(trainer, latent_loss + action_loss, critic_loss * scale))
    if global_count == 0:
        # Keep the VA-only optimizer update, but do not advance critic state
        # when every rank has zero legal transitions in this window.
        trainer.transformer.backward(latent_loss + action_loss)
        trainer.transformer.step()
        critic_norm = torch.zeros((), device=trainer.device)
    else:
        trainer.transformer.backward(latent_loss + action_loss + critic_loss * scale)
        parameters = sync_critic_gradients(trainer, world_size)
        critic_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=float(trainer.critic_config.max_grad_norm or float("inf")),
        )
        trainer.critic_optimizer.step()
        trainer.critic_bundle.update_target(trainer.critic_config.target_ema_rate)
        trainer.transformer.step()
    critic_metrics["critic_grad_norm"] = critic_norm.detach()
    critic_metrics["global_count"] = global_count
    return latent_loss.detach(), action_loss.detach(), critic_loss.detach(), critic_metrics


def run(args):
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    init_distributed(world_size, local_rank, rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    va_config = deepcopy(VA_CONFIGS[args.va_config_name])
    critic_config = CriticTrainingConfig.from_json(args.critic_config)
    va_config.batch_size = args.batch_size
    va_config.episode_subset_manifest = args.episode_subset_manifest
    va_config.save_root = args.save_root
    va_config.save_interval = args.save_interval
    va_config.log_freq = args.log_interval
    va_config.rank, va_config.local_rank, va_config.world_size = rank, local_rank, world_size
    va_config.use_deepspeed = bool(args.deepspeed)
    va_config.deepspeed_config = args.deepspeed_config
    if args.resume_from:
        va_config.resume_from = args.resume_from
    if args.critic_resume_from:
        object.__setattr__(critic_config, "resume_from", args.critic_resume_from)
    trainer = JointTrainer(va_config, critic_config, rank, local_rank, world_size, device)
    initialization = broadcast_critic_bundle_and_check(trainer, world_size, device)

    window_records, transition_summary = build_window_local_records(trainer.critic_dataset)
    schedule, schedule_summary = make_schedule(
        trainer.critic_dataset.base_dataset,
        world_size,
        args.batch_size,
        args.seed,
        set(window_records),
    )
    if args.max_steps > 0:
        schedule = schedule[:args.max_steps]
        schedule_summary["steps"] = len(schedule)
    resume_step = int(getattr(trainer, "step", 0))
    if resume_step < 0 or resume_step > len(schedule):
        raise RuntimeError(f"resume step {resume_step} is outside this deterministic schedule (0..{len(schedule)})")
    # The VA scheduler counts only the outer, shared-backbone updates.
    va_config.num_steps = len(schedule)
    trainer.va_config.num_steps = len(schedule)

    if rank == 0:
        trainer.save_dir.mkdir(parents=True, exist_ok=True)
        (trainer.save_dir / "windowsequential_summary.json").write_text(json.dumps({
            **transition_summary,
            **schedule_summary,
            **initialization,
            "batch_per_gpu": args.batch_size,
            "world_size": world_size,
            "va_window_steps": len(schedule),
            "critic_rule": (
                "for each window: chronological earlier legal transitions update the critic head; "
                "the time-last legal transition is combined once with that window's L_dyn and L_in"
            ),
            "critic_head_rounds": "all ranks synchronize to the maximum local chronological round; empty ranks contribute zero",
            "checkpoint_rule": f"every {args.save_interval} VA-window optimizer steps, plus final",
        }, indent=2) + "\n")
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])

    critic_updates = 0
    critic_transition_updates = 0
    for va_step, (_, entries) in enumerate(schedule[resume_step:], start=resume_step + 1):
        checkpoint_step = va_step % args.save_interval == 0 or va_step == len(schedule)
        local_entries = entries[rank * args.batch_size:(rank + 1) * args.batch_size]
        batch = default_collate([trainer.critic_dataset.base_dataset[index] for index, _ in local_entries])
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        local_rounds, final_pairs, local_total = split_window_transitions(
            window_records,
            trainer.critic_dataset.records,
            local_entries,
        )
        total_rounds = global_max_rounds(len(local_rounds), device)
        all_vectors = {"q": [], "value": [], "target": [], "td_abs": []}
        # One record per critic optimizer update.  These are later reduced as
        # a transition-count weighted global batch mean for honest monitoring.
        all_loss_records = []
        # Re-use a no-grad backbone read for every earlier transition.  These
        # head-only critic steps cannot alter VA/model weights.
        # Earlier critic-only rounds use the exact transformer feature path
        # used by the final joint round; only the gradient is detached.
        with torch.no_grad():
            frozen_inputs = trainer._prepare_va_input(batch)
            frozen_output = trainer.transformer(
                frozen_inputs, train_mode=True, return_features=True,
                critic_feature_layer=trainer.critic_config.feature_layer,
            )
            frozen_features = {key: value.detach() for key, value in frozen_output[3].items()}
        for round_index in range(total_rounds):
            pairs = local_rounds[round_index] if round_index < len(local_rounds) else []
            metadata = metadata_from_pairs(pairs, device)
            round_loss, round_metrics = critic_head_step(trainer, batch, frozen_features, metadata, world_size)
            if metadata is not None:
                transition_count = len(metadata["owner"])
                all_loss_records.append((
                    transition_count,
                    float(round_loss),
                    float(round_metrics["q_loss"]),
                    float(round_metrics["value_loss"]),
                ))
                for key in all_vectors: all_vectors[key].append(round_metrics[key])
            critic_updates += 1
            critic_transition_updates += len(pairs)

        final_metadata = metadata_from_pairs(final_pairs, device)
        latent_loss, action_loss, critic_loss, critic_metrics = joint_final_step(
            trainer, batch, final_metadata, world_size, measure_shared_grad=checkpoint_step
        )
        critic_updates += 1
        critic_transition_updates += len(final_pairs)
        if final_metadata is not None:
            transition_count = len(final_metadata["owner"])
            all_loss_records.append((
                transition_count,
                float(critic_loss),
                float(critic_metrics["q_loss"]),
                float(critic_metrics["value_loss"]),
            ))
            for key in all_vectors: all_vectors[key].append(critic_metrics[key])
        trainer.step = va_step

        should_log = va_step == 1 or va_step % args.log_interval == 0 or va_step == len(schedule)
        if should_log:
            stats = all_transition_stats(all_vectors, device)
            loss_averages = all_transition_loss_averages(all_loss_records, device)
            validation = fixed_window_validation(trainer, window_records, trainer.critic_dataset.records, world_size, args.batch_size, args.seed) if checkpoint_step else None
        if should_log and rank == 0:
            entry = {
                "va_window_step": va_step,
                "critic_optimizer_updates_rank0": critic_updates,
                "critic_transition_updates_rank0": critic_transition_updates,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "latent_loss_rank0": round(float(latent_loss), 6),
                "action_loss_rank0": round(float(action_loss), 6),
                "critic_loss_last_transition_rank0": round(float(critic_loss), 6),
                "q_loss_last_transition_rank0": round(float(critic_metrics["q_loss"]), 6),
                "value_loss_last_transition_rank0": round(float(critic_metrics["value_loss"]), 6),
                "all_transition_stats": stats,
                "critic_loss_all_transition_updates": loss_averages,
                "fixed_validation": validation,
                "critic_transitions_this_rank0_batch": local_total,
                "critic_head_rounds_this_window_batch": total_rounds,
                "critic_grad_norm_rank0": round(float(critic_metrics["critic_grad_norm"]), 6),
            }
            with trainer.log_file.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
        if va_step % args.save_interval == 0:
            trainer.save_checkpoint()
        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])

    trainer.save_checkpoint()
    if rank == 0:
        (trainer.save_dir / "FORMAL_EPOCH_COMPLETE").touch()
        (trainer.save_dir / "selection_stats.json").write_text(json.dumps({
            "va_window_steps": len(schedule),
            "critic_optimizer_updates": critic_updates,
            "critic_transition_updates": critic_transition_updates,
            "selection": "all chronological legal transitions; last one is the joint VA+critic transition",
        }, indent=2) + "\n")
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sequential_r2", choices=["sequential_r2"])
    parser.add_argument("--va-config-name", default="robocoin_droid_joint_actionaligned_r11_v1")
    parser.add_argument("--critic-config", required=True)
    parser.add_argument("--episode-subset-manifest", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed-config", required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means the full selected VA-window epoch")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--critic-resume-from", type=str, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
