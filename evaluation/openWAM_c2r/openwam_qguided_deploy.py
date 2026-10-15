"""Runtime Q-guidance adapter for the OpenWAM clean-to-random evaluation.

This keeps the upstream OpenWAM checkpoint loader/server and adds the IQL
critic trained with RL step 7266. The critic was trained on 16-action
transitions at feature_layer=0; only the first 16 actions of each 32-action
OpenWAM inference chunk are therefore scored/guided.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("openwam.q_guidance")


def _beta_envelope(sigmas: torch.Tensor, peak: float, concentration: float) -> torch.Tensor:
    sigma = sigmas.float().clamp(0.0, 1.0)
    shape = sigma.pow(peak * concentration) * (1.0 - sigma).pow((1.0 - peak) * concentration)
    total = shape.sum()
    if not torch.isfinite(total) or float(total) <= 1e-12:
        return torch.full_like(shape, 1.0 / max(1, shape.numel()))
    return shape / total


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


class OpenWAMQGuidance:
    def __init__(self, engine, critic_checkpoint: str):
        self.engine = engine
        self.arch = engine.architecture
        self.critic_checkpoint = critic_checkpoint
        self.chunk_size = int(os.environ.get("OPENWAM_Q_CHUNK_SIZE", "16"))
        self.feature_layer = int(os.environ.get("OPENWAM_Q_FEATURE_LAYER", "0"))
        self.interval = int(os.environ.get("OPENWAM_Q_GUIDANCE_INTERVAL", "5"))
        self.budget = float(os.environ.get("OPENWAM_Q_GUIDANCE_BUDGET", "0.08"))
        self.sigma_peak = float(os.environ.get("OPENWAM_Q_GUIDANCE_SIGMA_PEAK", "0.25"))
        self.sigma_concentration = float(os.environ.get("OPENWAM_Q_GUIDANCE_SIGMA_CONCENTRATION", "8.0"))
        if self.chunk_size <= 0 or self.interval <= 0 or self.budget < 0:
            raise ValueError("Q chunk size and interval must be positive; guidance budget must be non-negative")
        if not 0.0 < self.sigma_peak < 1.0 or self.sigma_concentration <= 0:
            raise ValueError("Invalid Q-guidance sigma envelope")
        if self.arch.critic is None:
            raise RuntimeError("The clean-SFT architecture has no critic module to receive RL step-7266 weights")
        self._load_critic()
        # Inference is frozen, but autograd remains enabled locally for
        # dQ/da because the generated candidate action itself requires grad.
        self.arch.requires_grad_(False)
        self.arch.eval()
        self._first_guidance_logged = False
        self.engine.action_velocity_guidance = self

    def _load_critic(self) -> None:
        from safetensors import safe_open

        state = {}
        with safe_open(self.critic_checkpoint, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key.startswith("critic."):
                    state[key[len("critic."):]] = checkpoint.get_tensor(key)
        if not state:
            raise RuntimeError(f"No critic.* weights found in {self.critic_checkpoint}")
        self.arch.critic.load_state_dict(state, strict=True)
        self.arch.critic.requires_grad_(False)
        self.arch.critic.eval()
        logger.info("Loaded %d critic tensors from %s", len(state), self.critic_checkpoint)

    def _prepare_state_inputs(self, prompt: str, image, proprio) -> dict:
        if image is None:
            raise RuntimeError("Q guidance requires the current first-frame image")
        if proprio is None:
            raise RuntimeError("Q guidance requires the normalized current proprio state")

        action_dim = int(self.arch.action_dim)
        action_indices = getattr(self.arch.normalizer, "_dst_index", None)
        active_dims = np.zeros(action_dim, dtype=bool)
        if action_indices is None:
            active_dims[:] = True
        else:
            active_dims[np.asarray(action_indices, dtype=np.int64)] = True

        state_indices = getattr(self.arch.normalizer, "_state_dst_index", action_indices)
        state_mask = np.zeros(action_dim, dtype=bool)
        if state_indices is None:
            state_mask[:] = True
        else:
            state_mask[np.asarray(state_indices, dtype=np.int64)] = True

        action_mask = np.broadcast_to(active_dims[None, :], (self.chunk_size, action_dim)).copy()
        state = _to_numpy(proprio).reshape(-1)
        sample = {
            # This mirrors openwam.rl.objective._state_sample: the current
            # observation is both the one-frame video and reference image.
            "video": [image],
            "first_frame_image": [image],
            "prompt": prompt or "",
            "action": np.zeros((self.chunk_size, action_dim), dtype=np.float32),
            "action_mask": action_mask,
            "proprio": state,
            "proprio_mask": state_mask.reshape(1, -1),
        }
        return self.arch.prepare_inputs([sample])

    def _step_weights(self, schedule) -> dict[int, float]:
        if schedule is None or len(schedule) < 2:
            raise RuntimeError("Q guidance requires OpenWAM's explicit action denoising schedule")
        num_train = float(getattr(self.arch.action_scheduler, "num_train_timesteps", 1000))
        selected = []
        selected_sigmas = []
        for index in range(len(schedule) - 1):
            sigma = float(schedule[index][1]) / num_train
            sigma_next = float(schedule[index + 1][1]) / num_train
            if index % self.interval == 0 and sigma > sigma_next:
                selected.append(index)
                selected_sigmas.append(sigma)
        if not selected:
            return {}
        envelope = _beta_envelope(
            torch.tensor(selected_sigmas, dtype=torch.float32),
            self.sigma_peak,
            self.sigma_concentration,
        )
        return {index: self.budget * float(weight) for index, weight in zip(selected, envelope)}

    def _q_gradient(self, clean_actions: torch.Tensor, context: dict) -> tuple[torch.Tensor, float, float]:
        candidate = clean_actions[:, : self.chunk_size, :].detach().clone().requires_grad_(True)
        if candidate.shape[1] != self.chunk_size:
            raise RuntimeError(
                f"Generated action chunk has {candidate.shape[1]} steps; the paired RL critic expects {self.chunk_size}"
            )
        inactive = context["inactive_dims"]
        if inactive is not None:
            candidate = candidate.clone()
            candidate[..., inactive] = 0
            candidate.requires_grad_(True)

        inputs = dict(context["prepared"])
        inputs.pop("actions", None)
        action_mask = ~inputs.pop("action_is_pad")
        token_mask = action_mask.any(dim=-1)
        proprio = inputs.pop("proprio", None)
        if "proprio_mask" in inputs:
            inputs["_proprio_sample_mask"] = inputs.pop("proprio_mask")
        inputs.pop("max_timestep_boundary", None)
        inputs.pop("min_timestep_boundary", None)
        inputs["latents"] = inputs["input_latents"]
        inputs["use_gradient_checkpointing"] = False
        inputs["use_gradient_checkpointing_offload"] = False
        timestep = torch.zeros(candidate.shape[0], device=candidate.device, dtype=candidate.dtype)

        with torch.enable_grad():
            features = self.arch.forward(
                candidate,
                timestep,
                proprio=proprio,
                timestep=timestep,
                feature_layer=self.feature_layer,
                action_token_mask=token_mask,
                **inputs,
            )
            critic = self.arch.critic
            q1 = critic.q1(features["state_tokens"], features["action_tokens"], action_mask=token_mask)
            q2 = critic.q2(features["state_tokens"], features["action_tokens"], action_mask=token_mask)
            q = torch.minimum(q1, q2).sum()
            q_value = float(q.detach().float().cpu())
            gradient = torch.autograd.grad(q, candidate, retain_graph=False, create_graph=False)[0]

        gradient = gradient.float()
        if inactive is not None:
            gradient[..., inactive] = 0
        norm = gradient.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
        if not torch.isfinite(gradient).all() or not torch.isfinite(norm).all():
            raise RuntimeError("Non-finite action gradient from the OpenWAM RL critic")
        gradient = gradient / norm.view(-1, 1, 1)
        full = torch.zeros_like(clean_actions, dtype=torch.float32)
        full[:, : self.chunk_size, :] = gradient
        return full.to(dtype=clean_actions.dtype), float(norm[0].detach().cpu()), q_value

    def begin_generation(
        self,
        architecture: Any,
        schedule: Any,
        prompt: str,
        first_frame_image: Any,
        proprio: Any,
    ) -> dict[str, Any]:
        """Prepare critic inputs once for the current denoising request."""
        if architecture is not self.arch:
            raise RuntimeError("Q guidance was attached to a different OpenWAM architecture")
        if float(getattr(self.engine, "_cfg_scale", 1.0)) != 1.0:
            raise RuntimeError("OpenWAM Q guidance currently requires CFG scale 1.0")
        image = first_frame_image[0] if first_frame_image else None
        prepared = self._prepare_state_inputs(prompt, image, proprio)
        inactive = self.arch._resolve_inactive_action_dims(None, self.arch.device)
        return {
            "prepared": prepared,
            "inactive_dims": inactive,
            "step_weights": self._step_weights(schedule),
        }

    def apply(
        self,
        *,
        context: dict[str, Any],
        step_index: int,
        sigma: float,
        action_latents: torch.Tensor,
        action_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Q guidance to one action velocity at its schedule index."""
        weight = context["step_weights"].get(step_index)
        if weight is None or weight == 0.0:
            return action_velocity

        sigma_value = float(sigma)
        clean = action_latents.detach() - action_velocity.detach() * sigma_value
        gradient, grad_norm, q_value = self._q_gradient(clean, context)
        # OpenWAM uses x_t=(1-sigma)x_0+sigma*x_1 and v=x_1-x_0.
        # Subtracting the ascent gradient from v moves the next x_t toward
        # higher-Q x_0 when sigma decreases.
        guided_velocity = action_velocity - gradient * float(weight)
        if not self._first_guidance_logged:
            logger.info(
                "First Q guidance applied: step=%d sigma=%.4f weight=%.6f q=%.6f grad_norm=%.6g",
                step_index,
                sigma_value,
                float(weight),
                q_value,
                grad_norm,
            )
            self._first_guidance_logged = True
        return guided_velocity


def main() -> None:
    checkpoint = os.environ.get("OPENWAM_Q_CHECKPOINT")
    if not checkpoint:
        raise SystemExit("OPENWAM_Q_CHECKPOINT must point to the paired RL critic checkpoint")

    from openwam.deploy import server

    def build_server_with_q_guidance(*args, **kwargs):
        policy_server = server.build_server_from_config(*args, **kwargs)
        OpenWAMQGuidance(policy_server.engine, checkpoint)
        return policy_server

    server.main(server_builder=build_server_with_q_guidance)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
