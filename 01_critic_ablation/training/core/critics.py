from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn


_LANGUAGE_PARAMETER_MARKERS = (
    ".text_proj.",
    ".language_query_norm.",
    ".language_context_norm.",
    ".language_attn.",
)


def is_language_critic_parameter(name: str) -> bool:
    return any(marker in name for marker in _LANGUAGE_PARAMETER_MARKERS)


@torch.no_grad()
def initialize_language_critic_parameters(module: nn.Module) -> None:
    """Initialize newly added language paths without changing inherited paths."""
    for child in module.modules():
        if isinstance(child, QueryScalarTransformerCritic):
            if child.text_proj is not None:
                nn.init.xavier_uniform_(child.text_proj.weight)
                if child.text_proj.bias is not None:
                    nn.init.zeros_(child.text_proj.bias)
        if not isinstance(child, QueryCrossAttentionBlock):
            continue
        if not child.language_conditioning:
            continue
        nn.init.ones_(child.language_query_norm.weight)
        nn.init.zeros_(child.language_query_norm.bias)
        nn.init.ones_(child.language_context_norm.weight)
        nn.init.zeros_(child.language_context_norm.bias)
        attention = child.language_attn
        if attention.in_proj_weight is not None:
            nn.init.xavier_uniform_(attention.in_proj_weight)
        else:
            nn.init.xavier_uniform_(attention.q_proj_weight)
            nn.init.xavier_uniform_(attention.k_proj_weight)
            nn.init.xavier_uniform_(attention.v_proj_weight)
        if attention.in_proj_bias is not None:
            nn.init.zeros_(attention.in_proj_bias)
        if attention.bias_k is not None:
            nn.init.zeros_(attention.bias_k)
        if attention.bias_v is not None:
            nn.init.zeros_(attention.bias_v)
        nn.init.zeros_(attention.out_proj.weight)
        if attention.out_proj.bias is not None:
            nn.init.zeros_(attention.out_proj.bias)


@torch.no_grad()
def copy_value_language_parameters_to_target(module: nn.Module) -> None:
    value = getattr(module, "value", None)
    target_value = getattr(module, "target_value", None)
    if value is None or target_value is None:
        return
    source_state = value.state_dict()
    target_state = target_value.state_dict()
    language_keys = {
        name for name in source_state if is_language_critic_parameter(f"value.{name}")
    }
    if not language_keys:
        raise ValueError("value critic has no language parameters to copy")
    missing = sorted(language_keys.difference(target_state))
    if missing:
        raise ValueError(f"target value is missing language parameters: {missing}")
    for name in language_keys:
        target_state[name].copy_(source_state[name])


class MaskedTokenPool(nn.Module):
    """Pool all valid tokens in one inference-sized latent/action chunk."""

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(
                f"tokens must have shape [B,F,T,D], got {tuple(tokens.shape)}"
            )
        token_mask = self._normalize_mask(mask, tokens)
        weights = token_mask.to(tokens.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=(1, 2)).clamp_min(1.0)
        return (tokens * weights).sum(dim=(1, 2)) / denominator

    @staticmethod
    def _normalize_mask(
        mask: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, token_count, _ = tokens.shape
        if mask.ndim == 5:
            # Action mask [B,C,F,N,1] -> token validity [B,F,N].
            mask = mask.squeeze(-1).any(dim=1)
        elif mask.ndim == 2:
            # Latent frame mask [B,F] -> all spatial context tokens in frame.
            mask = mask.unsqueeze(-1).expand(-1, -1, token_count)
        elif mask.ndim != 3:
            raise ValueError(
                "mask must have shape [B,C,F,N,1], [B,F,T], or [B,F]"
            )
        expected = (batch, frames, token_count)
        if tuple(mask.shape) != expected:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match {expected}"
            )
        return mask.bool()


class ScalarMLP(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        layers: list[nn.Module] = [nn.LayerNorm(feature_dim)]
        input_dim = feature_dim
        for _ in range(num_layers):
            layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.GELU()]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class TwinQCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.pool = MaskedTokenPool()
        self.q1 = ScalarMLP(feature_dim, hidden_dim, num_layers)
        self.q2 = ScalarMLP(feature_dim, hidden_dim, num_layers)

    def forward(
        self,
        action_tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool(action_tokens, action_mask).float()
        return self.q1(pooled), self.q2(pooled)


class ValueCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.pool = MaskedTokenPool()
        self.value = ScalarMLP(feature_dim, hidden_dim, num_layers)

    def forward(
        self,
        video_tokens: torch.Tensor,
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.value(
            self.pool(video_tokens, latent_mask).float()
        )


class CriticBundle(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        with_value: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.q = TwinQCritic(feature_dim, hidden_dim, num_layers)
        self.value = (
            ValueCritic(feature_dim, hidden_dim, num_layers)
            if with_value
            else None
        )
        self.target_value = (
            deepcopy(self.value).requires_grad_(False)
            if self.value is not None
            else None
        )

    @torch.no_grad()
    def update_target(self, rate: float) -> None:
        if self.value is None or self.target_value is None:
            return
        if not 0.0 <= rate <= 1.0:
            raise ValueError("EMA rate must be in [0, 1]")
        for target, source in zip(
            self.target_value.parameters(), self.value.parameters()
        ):
            target.lerp_(source, rate)


class QueryCrossAttentionBlock(nn.Module):
    """Transformer block for learned critic query tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        language_conditioning: bool = False,
    ) -> None:
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.language_conditioning = bool(language_conditioning)
        if self.language_conditioning:
            self.language_query_norm = nn.LayerNorm(hidden_dim)
            self.language_context_norm = nn.LayerNorm(hidden_dim)
            self.language_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        language_tokens: torch.Tensor | None = None,
        language_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_input = self.query_norm(query_tokens)
        query_update, _ = self.query_attn(
            query_input,
            query_input,
            query_input,
            need_weights=False,
        )
        query_tokens = query_tokens + query_update

        if self.language_conditioning:
            if language_tokens is None or language_mask is None:
                raise ValueError(
                    "language-conditioned critic requires language tokens and mask"
                )
            language_tokens, language_mask = _ensure_valid_context(
                language_tokens,
                language_mask,
            )
            language_update, _ = self.language_attn(
                self.language_query_norm(query_tokens),
                self.language_context_norm(language_tokens),
                self.language_context_norm(language_tokens),
                key_padding_mask=~language_mask.bool(),
                need_weights=False,
            )
            query_tokens = query_tokens + language_update

        context_tokens, context_mask = _ensure_valid_context(
            context_tokens,
            context_mask,
        )
        cross_update, _ = self.cross_attn(
            self.cross_norm(query_tokens),
            self.context_norm(context_tokens),
            self.context_norm(context_tokens),
            key_padding_mask=~context_mask.bool(),
            need_weights=False,
        )
        query_tokens = query_tokens + cross_update
        return query_tokens + self.ffn(self.ffn_norm(query_tokens))


class QueryScalarTransformerCritic(nn.Module):
    """Scalar critic using learnable query tokens over context tokens."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        num_query_tokens: int = 1,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        text_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_query_tokens <= 0:
            raise ValueError("num_query_tokens must be positive")
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.text_proj = (
            nn.Linear(text_embed_dim, hidden_dim)
            if text_embed_dim is not None
            else None
        )
        self.query_tokens = nn.Parameter(
            torch.randn(num_query_tokens, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.blocks = nn.ModuleList(
            [
                QueryCrossAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    language_conditioning=text_embed_dim is not None,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_tokens.ndim == 3:
            context_tokens = context_tokens.unsqueeze(1)
        if context_mask.ndim == 2:
            context_mask = context_mask.unsqueeze(1)
        if context_tokens.ndim != 4:
            raise ValueError(
                "context_tokens must have shape [B,L,S,D], got "
                f"{tuple(context_tokens.shape)}"
            )
        if context_mask.ndim != 3:
            raise ValueError(
                "context_mask must have shape [B,L,S], got "
                f"{tuple(context_mask.shape)}"
            )
        if context_mask.shape != context_tokens.shape[:3]:
            raise ValueError(
                "context_mask shape does not match context token sequence: "
                f"{tuple(context_mask.shape)} vs {tuple(context_tokens.shape[:3])}"
            )
        if context_tokens.shape[1] != len(self.blocks):
            raise ValueError(
                "context layer count must match query-transformer block count: "
                f"{context_tokens.shape[1]} != {len(self.blocks)}"
            )

        dtype = self.input_proj.weight.dtype
        context = self.input_proj(context_tokens.to(dtype))
        if self.text_proj is not None:
            if text_tokens is None or text_mask is None:
                raise ValueError(
                    "language-conditioned critic requires text tokens and mask"
                )
            language = self.text_proj(text_tokens.to(dtype))
            language_mask = text_mask.to(
                device=language.device,
                dtype=torch.bool,
            )
        else:
            language = None
            language_mask = None
        batch = context.shape[0]
        queries = self.query_tokens.to(dtype).unsqueeze(0).expand(batch, -1, -1)
        mask = context_mask.to(device=context.device, dtype=torch.bool)
        for layer_index, block in enumerate(self.blocks):
            queries = block(
                queries,
                context[:, layer_index],
                mask[:, layer_index],
                language,
                language_mask,
            )
        return self.head(self.norm(queries[:, -1])).squeeze(-1)


class TwinQueryTransformerQCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        num_query_tokens: int = 1,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        kwargs = dict(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_query_tokens=num_query_tokens,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.q1 = QueryScalarTransformerCritic(**kwargs)
        self.q2 = QueryScalarTransformerCritic(**kwargs)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1(context_tokens, context_mask),
            self.q2(context_tokens, context_mask),
        )


class QueryTransformerCriticBundle(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        with_value: bool = True,
        transformer_hidden_dim: int | None = None,
        transformer_num_layers: int | None = None,
        transformer_num_heads: int = 8,
        transformer_ffn_dim: int | None = None,
        transformer_dropout: float = 0.0,
        num_query_tokens: int = 1,
        **_: object,
    ) -> None:
        super().__init__()
        critic_hidden_dim = (
            hidden_dim if transformer_hidden_dim is None else transformer_hidden_dim
        )
        critic_num_layers = (
            num_layers if transformer_num_layers is None else transformer_num_layers
        )
        kwargs = dict(
            feature_dim=feature_dim,
            hidden_dim=critic_hidden_dim,
            num_layers=critic_num_layers,
            num_heads=transformer_num_heads,
            num_query_tokens=num_query_tokens,
            ffn_dim=transformer_ffn_dim,
            dropout=transformer_dropout,
        )
        self.q = TwinQueryTransformerQCritic(**kwargs)
        self.value = QueryScalarTransformerCritic(**kwargs) if with_value else None
        self.target_value = (
            deepcopy(self.value).requires_grad_(False)
            if self.value is not None
            else None
        )

    @torch.no_grad()
    def update_target(self, rate: float) -> None:
        if self.value is None or self.target_value is None:
            return
        if not 0.0 <= rate <= 1.0:
            raise ValueError("EMA rate must be in [0, 1]")
        for target, source in zip(
            self.target_value.parameters(), self.value.parameters()
        ):
            target.lerp_(source, rate)


class TwinLanguageQueryTransformerQCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        text_embed_dim: int = 4096,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        num_query_tokens: int = 1,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        kwargs = dict(
            feature_dim=feature_dim,
            text_embed_dim=text_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_query_tokens=num_query_tokens,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.q1 = QueryScalarTransformerCritic(**kwargs)
        self.q2 = QueryScalarTransformerCritic(**kwargs)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1(context_tokens, context_mask, text_tokens, text_mask),
            self.q2(context_tokens, context_mask, text_tokens, text_mask),
        )


class LanguageQueryTransformerCriticBundle(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        text_embed_dim: int = 4096,
        hidden_dim: int = 512,
        num_layers: int = 2,
        with_value: bool = True,
        transformer_hidden_dim: int | None = None,
        transformer_num_layers: int | None = None,
        transformer_num_heads: int = 8,
        transformer_ffn_dim: int | None = None,
        transformer_dropout: float = 0.0,
        num_query_tokens: int = 1,
        **_: object,
    ) -> None:
        super().__init__()
        critic_hidden_dim = (
            hidden_dim if transformer_hidden_dim is None else transformer_hidden_dim
        )
        critic_num_layers = (
            num_layers if transformer_num_layers is None else transformer_num_layers
        )
        kwargs = dict(
            feature_dim=feature_dim,
            text_embed_dim=text_embed_dim,
            hidden_dim=critic_hidden_dim,
            num_layers=critic_num_layers,
            num_heads=transformer_num_heads,
            num_query_tokens=num_query_tokens,
            ffn_dim=transformer_ffn_dim,
            dropout=transformer_dropout,
        )
        self.q = TwinLanguageQueryTransformerQCritic(**kwargs)
        self.value = QueryScalarTransformerCritic(**kwargs) if with_value else None
        self.target_value = (
            deepcopy(self.value).requires_grad_(False)
            if self.value is not None
            else None
        )

    @torch.no_grad()
    def update_target(self, rate: float) -> None:
        if self.value is None or self.target_value is None:
            return
        if not 0.0 <= rate <= 1.0:
            raise ValueError("EMA rate must be in [0, 1]")
        for target, source in zip(
            self.target_value.parameters(), self.value.parameters()
        ):
            target.lerp_(source, rate)


class ScalarFlowCrossAttentionBlock(nn.Module):
    """Single-token scalar-flow block with AdaLN, cross-attention, and FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4
        self.token_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    @staticmethod
    def _modulate(
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        token: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> torch.Tensor:
        (
            cross_shift,
            cross_scale,
            cross_gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        ) = self.ada(timestep_embedding).chunk(6, dim=-1)

        context_tokens, context_mask = _ensure_valid_context(
            context_tokens,
            context_mask,
        )
        cross_input = self._modulate(
            self.token_norm(token),
            cross_shift,
            cross_scale,
        )
        cross_update, _ = self.cross_attn(
            cross_input,
            self.context_norm(context_tokens),
            self.context_norm(context_tokens),
            key_padding_mask=~context_mask.bool(),
            need_weights=False,
        )
        token = token + cross_gate.unsqueeze(1) * cross_update

        ffn_input = self._modulate(
            self.ffn_norm(token),
            ffn_shift,
            ffn_scale,
        )
        return token + ffn_gate.unsqueeze(1) * self.ffn(ffn_input)


class ScalarFlowTransformerCritic(nn.Module):
    """Flow-matching scalar return head over layer-indexed context tokens."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.scalar_embed = nn.Linear(1, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                ScalarFlowCrossAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _sinusoidal_timestep(
        timestep: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        half = dim // 2
        if half == 0:
            return timestep[:, None]
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timestep.device, dtype=timestep.dtype)
            / max(half - 1, 1)
        )
        args = timestep[:, None] * freq[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        scalar_token: torch.Tensor,
        flow_timestep: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if context_tokens.ndim == 3:
            context_tokens = context_tokens.unsqueeze(1)
        if context_mask.ndim == 2:
            context_mask = context_mask.unsqueeze(1)
        if context_tokens.ndim != 4:
            raise ValueError(
                "context_tokens must have shape [B,L,S,D], got "
                f"{tuple(context_tokens.shape)}"
            )
        if context_mask.ndim != 3:
            raise ValueError(
                "context_mask must have shape [B,L,S], got "
                f"{tuple(context_mask.shape)}"
            )
        if context_mask.shape != context_tokens.shape[:3]:
            raise ValueError(
                "context_mask shape does not match context token sequence: "
                f"{tuple(context_mask.shape)} vs {tuple(context_tokens.shape[:3])}"
            )
        if context_tokens.shape[1] != len(self.blocks):
            raise ValueError(
                "context layer count must match scalar-flow block count: "
                f"{context_tokens.shape[1]} != {len(self.blocks)}"
            )

        dtype = self.input_proj.weight.dtype
        scalar_token = scalar_token.to(
            device=context_tokens.device,
            dtype=dtype,
        ).view(-1, 1)
        flow_timestep = flow_timestep.to(
            device=context_tokens.device,
            dtype=dtype,
        ).view(-1)
        context = self.input_proj(context_tokens.to(dtype))
        token = self.scalar_embed(scalar_token).unsqueeze(1)
        time_emb = self.time_embed(
            self._sinusoidal_timestep(flow_timestep, token.shape[-1])
        )
        mask = context_mask.to(device=context.device, dtype=torch.bool)
        for layer_index, block in enumerate(self.blocks):
            token = block(
                token,
                context[:, layer_index],
                mask[:, layer_index],
                time_emb,
            )
        return self.head(self.norm(token[:, -1])).squeeze(-1)


class FlowScalarQCriticEnsemble(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        num_q_heads: int = 2,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_q_heads <= 0:
            raise ValueError("num_q_heads must be positive")
        kwargs = dict(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.heads = nn.ModuleList(
            [ScalarFlowTransformerCritic(**kwargs) for _ in range(num_q_heads)]
        )

    def forward(
        self,
        scalar_token: torch.Tensor,
        flow_timestep: torch.Tensor,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        return [
            head(scalar_token, flow_timestep, context_tokens, context_mask)
            for head in self.heads
        ]


class FlowScalarTransformerCriticBundle(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        with_value: bool = True,
        transformer_hidden_dim: int | None = None,
        transformer_num_layers: int | None = None,
        transformer_num_heads: int = 8,
        transformer_ffn_dim: int | None = None,
        transformer_dropout: float = 0.0,
        num_q_heads: int = 2,
        **_: object,
    ) -> None:
        super().__init__()
        critic_hidden_dim = (
            hidden_dim if transformer_hidden_dim is None else transformer_hidden_dim
        )
        critic_num_layers = (
            num_layers if transformer_num_layers is None else transformer_num_layers
        )
        kwargs = dict(
            feature_dim=feature_dim,
            hidden_dim=critic_hidden_dim,
            num_layers=critic_num_layers,
            num_heads=transformer_num_heads,
            ffn_dim=transformer_ffn_dim,
            dropout=transformer_dropout,
        )
        self.q = FlowScalarQCriticEnsemble(num_q_heads=num_q_heads, **kwargs)
        self.value = ScalarFlowTransformerCritic(**kwargs) if with_value else None
        self.target_value = None

    @torch.no_grad()
    def update_target(self, rate: float) -> None:
        return


def _ensure_valid_context(
    context_tokens: torch.Tensor,
    context_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = context_mask.to(device=context_tokens.device, dtype=torch.bool)
    if mask.ndim != 2 or mask.shape != context_tokens.shape[:2]:
        raise ValueError("Invalid context mask shape")
    invalid_rows = ~mask.any(dim=1)
    if invalid_rows.any():
        context_tokens = context_tokens.clone()
        mask = mask.clone()
        context_tokens[invalid_rows, 0] = 0
        mask[invalid_rows, 0] = True
    return context_tokens, mask


CRITIC_REGISTRY = {
    "twin_mlp_v1": CriticBundle,
    "twin_query_transformer_v1": QueryTransformerCriticBundle,
    "twin_language_query_transformer_v1": LanguageQueryTransformerCriticBundle,
    "twin_flow_scalar_transformer_v1": FlowScalarTransformerCriticBundle,
}


def build_critic_bundle(
    critic_type: str,
    **kwargs,
) -> CriticBundle:
    try:
        constructor = CRITIC_REGISTRY[critic_type]
    except KeyError as error:
        raise ValueError(
            f"Unknown critic_type {critic_type!r}; "
            f"available: {sorted(CRITIC_REGISTRY)}"
        ) from error
    return constructor(**kwargs)
