"""
tests/test_model.py
====================
Unit tests for CleanUpAgent.  All tests run without MeltingPot or a GPU.
"""

import math

import pytest
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moltenpot.model import (
    MoltenpotAgent,
    CNNBackbone,
    _cnn_output_size,
    ORTH_GAIN,
    FC_UNITS,
    GRU_HIDDEN,
    NUM_ACTIONS,
)

# Alias for tests written against the old name
CleanUpAgent = MoltenpotAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def agent():
    return CleanUpAgent()


@pytest.fixture
def dummy_obs():
    """Random (B=2, T=5, 3, 88, 88) observation tensor in [0,1]."""
    return torch.rand(2, 5, 3, 88, 88)


@pytest.fixture
def dummy_h(agent):
    """Zero hidden state for batch_size=2."""
    return agent.initial_hidden(batch_size=2)   # (1, 2, 256)


# ---------------------------------------------------------------------------
# CNN geometry tests
# ---------------------------------------------------------------------------
class TestCNNGeometry:
    def test_cnn_output_size(self):
        """64 * 7 * 7 = 3136 for (88,88) input with the specified strides."""
        assert _cnn_output_size(88, 88) == 3136

    def test_cnn_backbone_output_shape(self):
        backbone = CNNBackbone()
        x = torch.rand(4, 3, 88, 88)
        out = backbone(x)
        assert out.shape == (4, FC_UNITS), f"Expected (4,{FC_UNITS}), got {out.shape}"


# ---------------------------------------------------------------------------
# Forward pass shape tests
# ---------------------------------------------------------------------------
class TestForwardPass:
    def test_logits_shape(self, agent, dummy_obs, dummy_h):
        logits, value, h_new = agent(dummy_obs, dummy_h)
        assert logits.shape == (2, 5, NUM_ACTIONS)

    def test_value_shape(self, agent, dummy_obs, dummy_h):
        _, value, _ = agent(dummy_obs, dummy_h)
        assert value.shape == (2, 5, 1)

    def test_h_state_shape(self, agent, dummy_obs, dummy_h):
        _, _, h_new = agent(dummy_obs, dummy_h)
        assert h_new.shape == (1, 2, GRU_HIDDEN)

    def test_h_state_changes(self, agent, dummy_obs, dummy_h):
        """The hidden state must be updated (non-zero output for non-zero input)."""
        _, _, h_new = agent(dummy_obs, dummy_h)
        # dummy_h is zeros; h_new should differ
        assert not torch.allclose(h_new, dummy_h), "h_state did not change"

    def test_h_detach_does_not_break_grad(self, agent, dummy_obs, dummy_h):
        """Detaching h at fragment boundary should not prevent loss.backward()."""
        _, _, h_new = agent(dummy_obs, dummy_h)
        h_detached = h_new.detach()

        obs2 = torch.rand(2, 5, 3, 88, 88)
        logits2, value2, _ = agent(obs2, h_detached)
        loss = value2.sum()
        loss.backward()       # must not raise


# ---------------------------------------------------------------------------
# Single-step act() helper
# ---------------------------------------------------------------------------
class TestActHelper:
    def test_act_returns_valid_action(self, agent):
        obs = torch.rand(3, 88, 88)
        h   = agent.initial_hidden(batch_size=1)
        action, log_prob, value, h_new = agent.act(obs, h)
        assert 0 <= action < NUM_ACTIONS
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
        assert h_new.shape == (1, 1, GRU_HIDDEN)

    def test_act_is_no_grad(self, agent):
        """act() should not create a computation graph."""
        obs = torch.rand(3, 88, 88)
        h   = agent.initial_hidden(batch_size=1)
        _, _, _, h_new = agent.act(obs, h)
        assert not h_new.requires_grad


class TestActBatch:
    """Tests for batched multi-agent inference."""

    @pytest.mark.parametrize("num_agents", [1, 3, 5, 7])
    def test_act_batch_output_shapes(self, agent, num_agents):
        import numpy as np
        obs = torch.rand(num_agents, 3, 88, 88)
        h   = agent.initial_hidden(batch_size=num_agents)
        actions, log_probs, values, h_new = agent.act_batch(obs, h)
        assert actions.shape == (num_agents,)
        assert log_probs.shape == (num_agents,)
        assert values.shape == (num_agents,)
        assert h_new.shape == (1, num_agents, GRU_HIDDEN)

    def test_act_batch_valid_actions(self, agent):
        import numpy as np
        obs = torch.rand(4, 3, 88, 88)
        h   = agent.initial_hidden(batch_size=4)
        actions, _, _, _ = agent.act_batch(obs, h)
        assert all(0 <= a < NUM_ACTIONS for a in actions)

    def test_act_batch_dtypes(self, agent):
        import numpy as np
        obs = torch.rand(3, 3, 88, 88)
        h   = agent.initial_hidden(batch_size=3)
        actions, log_probs, values, h_new = agent.act_batch(obs, h)
        assert actions.dtype == np.int32
        assert log_probs.dtype == np.float32
        assert values.dtype == np.float32
        assert not h_new.requires_grad


# ---------------------------------------------------------------------------
# Orthogonal initialisation tests
# ---------------------------------------------------------------------------
class TestOrthogonalInit:
    def test_conv_weight_is_orthogonal(self, agent):
        """For a weight matrix W, W @ W.T ≈ I (up to scale)."""
        conv_weight = agent.backbone.conv_net[0].weight  # Conv2d → (32, 3, 8, 8)
        W = conv_weight.view(conv_weight.shape[0], -1)   # (32, 192)
        # orthogonal ⟹ W @ W.T = scale * I
        product = W @ W.T  # (32, 32)
        diag    = torch.diag(product)
        off_diag = product - torch.diag(diag)
        # Off-diagonal elements should be small
        assert off_diag.abs().max().item() < 1e-4, \
            "Conv weight is not approximately orthogonal"

    def test_biases_zero(self, agent):
        for name, param in agent.named_parameters():
            if "bias" in name:
                assert param.abs().max().item() < 1e-7, \
                    f"Bias {name} is not zero after init"

    def test_actor_logits_small_initially(self, agent):
        """Actor head uses gain=0.01 → initial logits very close to zero."""
        obs = torch.rand(1, 1, 3, 88, 88)
        h   = agent.initial_hidden(1)
        logits, _, _ = agent(obs, h)
        # All logits should be very small (< 0.1 magnitude)
        assert logits.abs().max().item() < 0.1, \
            f"Initial actor logits too large: {logits.abs().max().item():.4f}"


# ---------------------------------------------------------------------------
# Weight serialisation (Ray transfer simulation)
# ---------------------------------------------------------------------------
class TestWeightTransfer:
    def test_get_set_weights_roundtrip(self, agent):
        weights = agent.get_weights()
        fresh   = CleanUpAgent()
        fresh.set_weights(weights)
        for (k1, v1), (k2, v2) in zip(
            agent.state_dict().items(), fresh.state_dict().items()
        ):
            assert torch.allclose(v1, v2), f"Weight mismatch for key {k1}"

    def test_weights_on_cpu(self, agent):
        weights = agent.get_weights()
        for k, v in weights.items():
            assert v.device.type == "cpu", f"Weight {k} is not on CPU"
