"""
Attention-Based Coupling Policy Network
========================================
Implements the learned inter-organ coupling equation for the cardiorenal
digital twin. The attention architecture encodes the inductive bias that
inter-organ coupling is about *which outputs of one organ are relevant
to the other* — precisely what cross-attention computes.

Architecture:
    cardiac_obs (B,12) → OrganEncoder → (B,12,D) cardiac tokens
    renal_obs (B,10)   → OrganEncoder → (B,10,D) renal tokens
    meta+temporal (B,10) → Linear → (B,1,D) context token

    cardiac_tokens ←cross-attn→ renal_tokens  (n_cross_layers)

    pool → CouplingPolicyHead → (alphas, residuals, value, attn_weights)

The cross-attention weights are directly interpretable:
    h_cross_r_attn[i,j] = how much cardiac feature i depends on renal feature j
    r_cross_h_attn[i,j] = how much renal feature i depends on cardiac feature j
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional


class FeatureTokenizer(nn.Module):
    """Project each scalar feature to a D-dim token with positional embedding.

    Each input feature gets its own learned linear projection (1 → D) plus
    a positional embedding that encodes feature identity (e.g., "this token
    represents MAP" vs "this token represents CO").
    """

    def __init__(self, n_features: int, embed_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        # Per-feature projections: each scalar → D-dim
        self.projections = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(n_features)
        ])
        # Positional embeddings encode feature identity
        self.pos_embed = nn.Parameter(torch.randn(n_features, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_features)

        Returns
        -------
        tokens : (B, n_features, D)
        """
        B = x.shape[0]
        tokens = torch.stack([
            self.projections[i](x[:, i:i+1])  # (B, D)
            for i in range(self.n_features)
        ], dim=1)  # (B, n_features, D)
        tokens = tokens + self.pos_embed.unsqueeze(0)  # broadcast positional
        return tokens


class SelfAttentionBlock(nn.Module):
    """Within-organ multi-head self-attention with pre-norm residual."""

    def __init__(self, embed_dim: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, N, D)

        Returns
        -------
        out : (B, N, D)
        attn_weights : (B, N, N) — averaged across heads
        """
        normed = self.norm(x)
        attn_out, attn_weights = self.attn(normed, normed, normed)
        return x + attn_out, attn_weights


class CrossAttentionBlock(nn.Module):
    """Inter-organ cross-attention: Q from organ A, K/V from organ B."""

    def __init__(self, embed_dim: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
        )

    def forward(
        self, query_tokens: torch.Tensor, context_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query_tokens : (B, N_q, D) — organ being modulated
        context_tokens : (B, N_kv, D) — organ providing signals

        Returns
        -------
        out : (B, N_q, D)
        cross_weights : (B, N_q, N_kv) — averaged across heads
        """
        q = self.norm_q(query_tokens)
        kv = self.norm_kv(context_tokens)
        attn_out, cross_weights = self.cross_attn(q, kv, kv)
        return query_tokens + attn_out, cross_weights


class OrganEncoder(nn.Module):
    """Encode one organ's features: tokenize → self-attention."""

    def __init__(self, n_features: int, embed_dim: int = 64,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, embed_dim)
        self.self_attn = SelfAttentionBlock(embed_dim, n_heads, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, n_features)

        Returns
        -------
        tokens : (B, n_features, D) — contextualized tokens
        self_attn_weights : (B, n_features, n_features)
        """
        tokens = self.tokenizer(x)
        tokens, self_attn_wt = self.self_attn(tokens)
        return tokens, self_attn_wt


class CouplingPolicyHead(nn.Module):
    """Decode attended tokens → coupling weights + residuals + value.

    Outputs:
        alphas: 5-dim coupling weights bounded to [alpha_min, alpha_max]
        residuals: 10-dim inflammatory corrections bounded to [res_min, res_max]
        value: scalar state value for PPO critic
    """

    def __init__(self, embed_dim: int = 64, n_coupling: int = 5,
                 n_residuals: int = 10,
                 alpha_min: float = 0.5, alpha_max: float = 1.5,
                 residual_min: float = -0.3, residual_max: float = 0.3):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.residual_min = residual_min
        self.residual_max = residual_max

        # Shared trunk from pooled embeddings
        self.trunk = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Alpha head: mean + log_std for Gaussian exploration
        self.alpha_mean = nn.Linear(embed_dim, n_coupling)
        self.alpha_logstd = nn.Parameter(torch.zeros(n_coupling) - 0.5)

        # Residual head: mean + log_std
        self.residual_mean = nn.Linear(embed_dim, n_residuals)
        self.residual_logstd = nn.Parameter(torch.zeros(n_residuals) - 1.0)

        # Value head (critic)
        self.value_head = nn.Linear(embed_dim, 1)

    def forward(self, h_pooled: torch.Tensor, r_pooled: torch.Tensor):
        """
        Parameters
        ----------
        h_pooled : (B, D) — pooled cardiac embedding
        r_pooled : (B, D) — pooled renal embedding

        Returns
        -------
        alpha_mean : (B, 5) — coupling weight means (bounded)
        alpha_std : (B, 5) — coupling weight stds
        residual_mean : (B, 10) — residual correction means (bounded)
        residual_std : (B, 10) — residual correction stds
        value : (B, 1)
        """
        joint = self.trunk(torch.cat([h_pooled, r_pooled], dim=-1))

        # Alpha: sigmoid → [alpha_min, alpha_max]
        alpha_raw = self.alpha_mean(joint)
        alpha_mean = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(alpha_raw)
        alpha_std = self.alpha_logstd.exp().expand_as(alpha_mean)

        # Residual: tanh → [residual_min, residual_max]
        residual_raw = self.residual_mean(joint)
        residual_mean = 0.5 * (self.residual_min + self.residual_max) + \
                        0.5 * (self.residual_max - self.residual_min) * torch.tanh(residual_raw)
        residual_std = self.residual_logstd.exp().expand_as(residual_mean)

        value = self.value_head(joint)

        return alpha_mean, alpha_std, residual_mean, residual_std, value


class AttentionCouplingPolicy(nn.Module):
    """Full RL policy: organ observations → attention → coupling equation.

    This is the learned coupling equation f(h_t, r_t, d_t, i_t) → (α_vec, Δϕ)
    that sits between CircAdapt and Hallow, producing the per-message weights
    and diabetes modifier corrections that define how the two organ models
    interact at each time step.

    The cross-attention weights are the learned coefficients of this equation,
    directly interpretable as "how much does cardiac feature X influence renal
    feature Y at this disease state."
    """

    def __init__(
        self,
        cardiac_dim: int = 12,
        renal_dim: int = 10,
        meta_dim: int = 5,
        temporal_dim: int = 5,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_cross_layers: int = 2,
        n_coupling: int = 5,
        n_residuals: int = 10,
        dropout: float = 0.1,
        alpha_min: float = 0.5,
        alpha_max: float = 1.5,
        residual_min: float = -0.3,
        residual_max: float = 0.3,
    ):
        super().__init__()
        self.cardiac_dim = cardiac_dim
        self.renal_dim = renal_dim
        self.meta_dim = meta_dim
        self.temporal_dim = temporal_dim
        self.embed_dim = embed_dim

        # Organ encoders (tokenize + self-attention)
        self.cardiac_encoder = OrganEncoder(cardiac_dim, embed_dim, n_heads, dropout)
        self.renal_encoder = OrganEncoder(renal_dim, embed_dim, n_heads, dropout)

        # Meta + temporal context projection → single context token
        self.context_proj = nn.Sequential(
            nn.Linear(meta_dim + temporal_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Bidirectional cross-attention layers
        self.h_cross_r_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])
        self.r_cross_h_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # Policy and value heads
        self.policy_head = CouplingPolicyHead(
            embed_dim, n_coupling, n_residuals,
            alpha_min, alpha_max, residual_min, residual_max,
        )

        # Store last attention weights for interpretability
        self._last_attn_weights = {}

    def forward(
        self,
        cardiac_obs: torch.Tensor,
        renal_obs: torch.Tensor,
        meta_obs: torch.Tensor,
        temporal_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass of the coupling equation.

        Parameters
        ----------
        cardiac_obs : (B, 12) — cardiac hemodynamic features
        renal_obs : (B, 10) — renal state features
        meta_obs : (B, 5) — disease/metabolic context
        temporal_obs : (B, 5) — temporal features (progress, deltas)

        Returns
        -------
        alpha_mean : (B, 5) — per-message coupling weight means
        alpha_std : (B, 5) — coupling weight exploration noise
        residual_mean : (B, 10) — inflammatory residual means
        residual_std : (B, 10) — residual exploration noise
        value : (B, 1) — state value estimate
        attn_weights : dict — attention maps for interpretability
        """
        # 1. Encode organ features → contextualized tokens
        h_tokens, h_self_wt = self.cardiac_encoder(cardiac_obs)   # (B, 12, D)
        r_tokens, r_self_wt = self.renal_encoder(renal_obs)       # (B, 10, D)

        # 2. Create context token from meta + temporal
        context = self.context_proj(
            torch.cat([meta_obs, temporal_obs], dim=-1)
        ).unsqueeze(1)  # (B, 1, D)

        # 3. Append context token to both organ token sequences
        h_tokens = torch.cat([h_tokens, context], dim=1)  # (B, 13, D)
        r_tokens = torch.cat([r_tokens, context], dim=1)   # (B, 11, D)

        # 4. Bidirectional cross-attention
        h_cross_wts = []
        r_cross_wts = []
        for h_cross_r, r_cross_h in zip(self.h_cross_r_layers, self.r_cross_h_layers):
            h_tokens, h_cw = h_cross_r(h_tokens, r_tokens)
            r_tokens, r_cw = r_cross_h(r_tokens, h_tokens)
            h_cross_wts.append(h_cw)
            r_cross_wts.append(r_cw)

        # 5. Pool tokens to fixed-size embeddings
        h_pooled = h_tokens.mean(dim=1)  # (B, D)
        r_pooled = r_tokens.mean(dim=1)  # (B, D)

        # 6. Policy + value heads
        alpha_mean, alpha_std, res_mean, res_std, value = \
            self.policy_head(h_pooled, r_pooled)

        # Store attention weights for interpretability
        attn_weights = {
            'cardiac_self': h_self_wt,
            'renal_self': r_self_wt,
            'h_cross_r': h_cross_wts,  # list of (B, N_h, N_r) per layer
            'r_cross_h': r_cross_wts,  # list of (B, N_r, N_h) per layer
        }
        self._last_attn_weights = attn_weights

        return alpha_mean, alpha_std, res_mean, res_std, value, attn_weights

    def get_action(
        self,
        obs_dict: Dict[str, float],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """Convenience method: observation dict → sampled action.

        Parameters
        ----------
        obs_dict : dict
            Observation with named features (32 keys).
        deterministic : bool
            If True, use mean action (no exploration noise).

        Returns
        -------
        action : np.ndarray (15,) — [alpha_5, residual_10]
        log_prob : float — log probability of the sampled action
        value : float — state value estimate
        """
        from config import (CARDIAC_FEATURE_NAMES, RENAL_FEATURE_NAMES,
                           META_FEATURE_NAMES, TEMPORAL_FEATURE_NAMES)

        # Split observation into organ groups, replacing NaN with 0.0
        def _safe_val(v):
            f = float(v)
            return 0.0 if (f != f) else f  # NaN != NaN

        cardiac = torch.tensor(
            [_safe_val(obs_dict[k]) for k in CARDIAC_FEATURE_NAMES], dtype=torch.float32,
        ).unsqueeze(0)
        renal = torch.tensor(
            [_safe_val(obs_dict[k]) for k in RENAL_FEATURE_NAMES], dtype=torch.float32,
        ).unsqueeze(0)
        meta = torch.tensor(
            [_safe_val(obs_dict[k]) for k in META_FEATURE_NAMES], dtype=torch.float32,
        ).unsqueeze(0)
        temporal = torch.tensor(
            [_safe_val(obs_dict[k]) for k in TEMPORAL_FEATURE_NAMES], dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            a_mean, a_std, r_mean, r_std, value, _ = self.forward(
                cardiac, renal, meta, temporal,
            )

        if deterministic:
            alpha_action = a_mean.squeeze(0).numpy()
            res_action = r_mean.squeeze(0).numpy()
            # Compute log_prob at mean (= 0 for Normal at mean, but compute properly)
            alpha_dist = Normal(a_mean, a_std)
            res_dist = Normal(r_mean, r_std)
            log_prob = (alpha_dist.log_prob(a_mean).sum() +
                       res_dist.log_prob(r_mean).sum()).item()
        else:
            alpha_dist = Normal(a_mean, a_std)
            res_dist = Normal(r_mean, r_std)
            alpha_action = alpha_dist.sample().squeeze(0).numpy()
            res_action = res_dist.sample().squeeze(0).numpy()
            log_prob = (alpha_dist.log_prob(torch.from_numpy(alpha_action).unsqueeze(0)).sum() +
                       res_dist.log_prob(torch.from_numpy(res_action).unsqueeze(0)).sum()).item()

        # Clamp to valid bounds
        alpha_action = np.clip(alpha_action, self.policy_head.alpha_min,
                              self.policy_head.alpha_max)
        res_action = np.clip(res_action, self.policy_head.residual_min,
                            self.policy_head.residual_max)

        action = np.concatenate([alpha_action, res_action])
        return action, log_prob, value.item()

    def evaluate_actions(
        self,
        cardiac_obs: torch.Tensor,
        renal_obs: torch.Tensor,
        meta_obs: torch.Tensor,
        temporal_obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities, entropy, and values for given actions.

        Used during PPO update to compute the importance sampling ratio.

        Parameters
        ----------
        cardiac_obs : (B, 12)
        renal_obs : (B, 10)
        meta_obs : (B, 5)
        temporal_obs : (B, 5)
        actions : (B, 15) — [alpha_5, residual_10]

        Returns
        -------
        log_probs : (B,) — log probability of each action
        entropy : (B,) — entropy of the action distribution
        values : (B, 1) — state value estimates
        """
        a_mean, a_std, r_mean, r_std, values, _ = self.forward(
            cardiac_obs, renal_obs, meta_obs, temporal_obs,
        )

        alpha_actions = actions[:, :5]
        residual_actions = actions[:, 5:]

        alpha_dist = Normal(a_mean, a_std)
        residual_dist = Normal(r_mean, r_std)

        alpha_log_prob = alpha_dist.log_prob(alpha_actions).sum(dim=-1)
        residual_log_prob = residual_dist.log_prob(residual_actions).sum(dim=-1)
        log_probs = alpha_log_prob + residual_log_prob

        alpha_entropy = alpha_dist.entropy().sum(dim=-1)
        residual_entropy = residual_dist.entropy().sum(dim=-1)
        entropy = alpha_entropy + residual_entropy

        return log_probs, entropy, values

    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Return the most recent attention weight matrices.

        Useful for interpretability: examining which cardiac features
        attend to which renal features (and vice versa).
        """
        return self._last_attn_weights

    def get_config(self) -> Dict:
        """Return constructor arguments for checkpoint serialization."""
        return {
            'cardiac_dim': self.cardiac_dim,
            'renal_dim': self.renal_dim,
            'meta_dim': self.meta_dim,
            'temporal_dim': self.temporal_dim,
            'embed_dim': self.embed_dim,
            'n_heads': self.cardiac_encoder.self_attn.attn.num_heads,
            'n_cross_layers': len(self.h_cross_r_layers),
            'n_coupling': self.policy_head.alpha_mean.out_features,
            'n_residuals': self.policy_head.residual_mean.out_features,
            'alpha_min': self.policy_head.alpha_min,
            'alpha_max': self.policy_head.alpha_max,
            'residual_min': self.policy_head.residual_min,
            'residual_max': self.policy_head.residual_max,
        }
