"""Unit tests for the attention-based coupling policy network."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import tempfile
import torch
import numpy as np
from models.attention_coupling import (
    FeatureTokenizer, SelfAttentionBlock, CrossAttentionBlock,
    OrganEncoder, CouplingPolicyHead, AttentionCouplingPolicy,
)


@pytest.fixture
def policy():
    """Create a default policy for testing."""
    return AttentionCouplingPolicy(
        cardiac_dim=12, renal_dim=10, meta_dim=5, temporal_dim=5,
        embed_dim=64, n_heads=4, n_cross_layers=2,
    )


@pytest.fixture
def sample_inputs():
    """Create sample batch inputs."""
    B = 4
    return {
        'cardiac': torch.randn(B, 12),
        'renal': torch.randn(B, 10),
        'meta': torch.randn(B, 5),
        'temporal': torch.randn(B, 5),
    }


class TestFeatureTokenizer:
    def test_output_shape(self):
        tok = FeatureTokenizer(n_features=12, embed_dim=64)
        x = torch.randn(3, 12)
        tokens = tok(x)
        assert tokens.shape == (3, 12, 64)

    def test_different_feature_counts(self):
        for n in [5, 10, 15]:
            tok = FeatureTokenizer(n_features=n, embed_dim=32)
            tokens = tok(torch.randn(2, n))
            assert tokens.shape == (2, n, 32)


class TestSelfAttentionBlock:
    def test_output_shape(self):
        block = SelfAttentionBlock(embed_dim=64, n_heads=4)
        x = torch.randn(2, 12, 64)
        out, weights = block(x)
        assert out.shape == x.shape
        assert weights.shape[0] == 2  # batch dim

    def test_residual_connection(self):
        """Output should differ from input (attention modifies it)."""
        block = SelfAttentionBlock(embed_dim=64, n_heads=4)
        x = torch.randn(1, 8, 64)
        out, _ = block(x)
        # They shouldn't be exactly equal (attention adds information)
        assert not torch.allclose(out, x, atol=1e-6)


class TestCrossAttentionBlock:
    def test_output_shape(self):
        block = CrossAttentionBlock(embed_dim=64, n_heads=4)
        q = torch.randn(2, 12, 64)
        kv = torch.randn(2, 10, 64)
        out, weights = block(q, kv)
        assert out.shape == q.shape
        assert weights.shape == (2, 12, 10)


class TestAttentionCouplingPolicy:

    def test_forward_shapes(self, policy, sample_inputs):
        """All outputs should have correct shapes."""
        a_mean, a_std, r_mean, r_std, value, attn = policy(
            sample_inputs['cardiac'], sample_inputs['renal'],
            sample_inputs['meta'], sample_inputs['temporal'],
        )
        B = sample_inputs['cardiac'].shape[0]
        assert a_mean.shape == (B, 5)
        assert a_std.shape == (B, 5)
        assert r_mean.shape == (B, 10)
        assert r_std.shape == (B, 10)
        assert value.shape == (B, 1)

    def test_alpha_bounds(self, policy, sample_inputs):
        """Alpha means should be within [0.5, 1.5]."""
        a_mean, _, _, _, _, _ = policy(
            sample_inputs['cardiac'], sample_inputs['renal'],
            sample_inputs['meta'], sample_inputs['temporal'],
        )
        assert (a_mean >= 0.5 - 0.01).all()  # small tolerance for numerical
        assert (a_mean <= 1.5 + 0.01).all()

    def test_residual_bounds(self, policy, sample_inputs):
        """Residual means should be within [-0.3, 0.3]."""
        _, _, r_mean, _, _, _ = policy(
            sample_inputs['cardiac'], sample_inputs['renal'],
            sample_inputs['meta'], sample_inputs['temporal'],
        )
        assert (r_mean >= -0.3 - 0.01).all()
        assert (r_mean <= 0.3 + 0.01).all()

    def test_std_positive(self, policy, sample_inputs):
        """Standard deviations should be positive."""
        _, a_std, _, r_std, _, _ = policy(
            sample_inputs['cardiac'], sample_inputs['renal'],
            sample_inputs['meta'], sample_inputs['temporal'],
        )
        assert (a_std > 0).all()
        assert (r_std > 0).all()

    def test_attention_weights_returned(self, policy, sample_inputs):
        """Attention weight dicts should contain expected keys."""
        _, _, _, _, _, attn = policy(
            sample_inputs['cardiac'], sample_inputs['renal'],
            sample_inputs['meta'], sample_inputs['temporal'],
        )
        assert 'cardiac_self' in attn
        assert 'renal_self' in attn
        assert 'h_cross_r' in attn
        assert 'r_cross_h' in attn
        assert len(attn['h_cross_r']) == 2  # n_cross_layers=2

    def test_gradient_flow(self, policy, sample_inputs):
        """Parameters in the forward path should receive gradients."""
        a_mean, a_std, r_mean, r_std, value, _ = policy(
            sample_inputs['cardiac'], sample_inputs['renal'],
            sample_inputs['meta'], sample_inputs['temporal'],
        )
        # Include std in loss so logstd params get gradients too
        loss = a_mean.sum() + r_mean.sum() + value.sum() + a_std.sum() + r_std.sum()
        loss.backward()

        for name, param in policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_deterministic_eval_mode(self, policy, sample_inputs):
        """Same input should produce same output in eval mode."""
        policy.eval()
        with torch.no_grad():
            out1 = policy(sample_inputs['cardiac'], sample_inputs['renal'],
                         sample_inputs['meta'], sample_inputs['temporal'])
            out2 = policy(sample_inputs['cardiac'], sample_inputs['renal'],
                         sample_inputs['meta'], sample_inputs['temporal'])
        assert torch.allclose(out1[0], out2[0])  # alpha_mean
        assert torch.allclose(out1[2], out2[2])  # residual_mean
        assert torch.allclose(out1[4], out2[4])  # value

    def test_evaluate_actions_shapes(self, policy, sample_inputs):
        """evaluate_actions should return correct shapes."""
        B = sample_inputs['cardiac'].shape[0]
        actions = torch.randn(B, 15)

        log_probs, entropy, values = policy.evaluate_actions(
            sample_inputs['cardiac'], sample_inputs['renal'],
            sample_inputs['meta'], sample_inputs['temporal'],
            actions,
        )
        assert log_probs.shape == (B,)
        assert entropy.shape == (B,)
        assert values.shape == (B, 1)

    def test_save_load_roundtrip(self, policy, sample_inputs):
        """Save and reload should produce identical outputs."""
        policy.eval()
        with torch.no_grad():
            original_out = policy(
                sample_inputs['cardiac'], sample_inputs['renal'],
                sample_inputs['meta'], sample_inputs['temporal'],
            )

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            tmppath = f.name

        try:
            torch.save({
                'model_state_dict': policy.state_dict(),
                'config': policy.get_config(),
            }, tmppath)

            # Reload
            ckpt = torch.load(tmppath, map_location='cpu', weights_only=False)
            loaded = AttentionCouplingPolicy(**ckpt['config'])
            loaded.load_state_dict(ckpt['model_state_dict'])
            loaded.eval()

            with torch.no_grad():
                loaded_out = loaded(
                    sample_inputs['cardiac'], sample_inputs['renal'],
                    sample_inputs['meta'], sample_inputs['temporal'],
                )

            assert torch.allclose(original_out[0], loaded_out[0], atol=1e-6)
            assert torch.allclose(original_out[2], loaded_out[2], atol=1e-6)
        finally:
            os.unlink(tmppath)

    def test_get_config_roundtrip(self, policy):
        """get_config should return valid constructor kwargs."""
        config = policy.get_config()
        new_policy = AttentionCouplingPolicy(**config)
        # Should have same number of parameters
        orig_params = sum(p.numel() for p in policy.parameters())
        new_params = sum(p.numel() for p in new_policy.parameters())
        assert orig_params == new_params

    def test_get_action_shapes(self, policy):
        """get_action should return (15,), scalar, scalar."""
        obs_dict = {}
        from config import (CARDIAC_FEATURE_NAMES, RENAL_FEATURE_NAMES,
                           META_FEATURE_NAMES, TEMPORAL_FEATURE_NAMES)
        for name in CARDIAC_FEATURE_NAMES + RENAL_FEATURE_NAMES + META_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES:
            obs_dict[name] = np.random.randn()

        policy.eval()
        action, log_prob, value = policy.get_action(obs_dict, deterministic=True)
        assert action.shape == (15,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        # Alphas in bounds
        assert np.all(action[:5] >= 0.5 - 0.01)
        assert np.all(action[:5] <= 1.5 + 0.01)
