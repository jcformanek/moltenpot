"""
model.py — MoltenpotAgent
=========================
Generic recurrent Actor-Critic network for MeltingPot substrates.

Architecture
------------
Input: (B, T, 3, 88, 88)  — B=batch, T=time, CHW visual observation

CNN backbone (shared):
    Conv2d(3,  32, k=8, s=4) → ReLU   → (B·T, 32, 20, 20)
    Conv2d(32, 64, k=4, s=2) → ReLU   → (B·T, 64,  9,  9)
    Conv2d(64, 64, k=3, s=1) → ReLU   → (B·T, 64,  7,  7)
    Flatten                             → (B·T, 3136)
    Linear(3136, 256) + ReLU           → (B·T, 256)

Temporal head:
    GRU(256, 256, batch_first=True)    → (B, T, 256)

Output heads:
    actor  → Linear(256, num_actions)  → logits
    critic → Linear(256, 1)            → value scalar
    q1, q2 → Linear(256, num_actions)  → Q-values (offline RL)

Initialization: Orthogonal, gain=√2 for all weight matrices, bias=0.
The GRU input+hidden weights also use orthogonal init.

Notes on hidden state
---------------------
The hidden state tensor `h` has shape (num_layers=1, B, H).
Callers MUST detach `h` at fragment boundaries:
    h = h.detach()
This prevents gradients from flowing into the previous fragment,
which would OOM on long rollouts and invalidate the PPO objective.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Architecture defaults
# ---------------------------------------------------------------------------
CNN_CHANNELS  = (3, 32, 64, 64)
CNN_KERNELS   = (8, 4, 3)
CNN_STRIDES   = (4, 2, 1)
FC_UNITS      = 256
GRU_HIDDEN    = 256
NUM_ACTIONS   = 9   # Clean Up default; override via constructor
MAX_AGENTS    = 8   # one-hot agent-ID dimensionality; must be ≥ num focal agents
ORTH_GAIN     = math.sqrt(2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cnn_output_size(h: int = 88, w: int = 88) -> int:
    """Return the flattened size after the three conv layers."""
    for k, s in zip(CNN_KERNELS, CNN_STRIDES):
        h = (h - k) // s + 1
        w = (w - k) // s + 1
    return CNN_CHANNELS[-1] * h * w   # 64 * 7 * 7 = 3136 for 88×88 input


def _orthogonal_init(module: nn.Module, gain: float = ORTH_GAIN) -> None:
    """
    Apply orthogonal initialisation to all weight matrices in *module*
    and zero out all biases.  Works for Linear, Conv2d, and GRU layers.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight" in name:
                for row in param.chunk(3, dim=0):   # 3 gates
                    nn.init.orthogonal_(row, gain=gain)
            elif "bias" in name:
                nn.init.zeros_(param)


# ---------------------------------------------------------------------------
# CNN Backbone
# ---------------------------------------------------------------------------
class CNNBackbone(nn.Module):
    """
    Three-layer convolutional feature extractor.

    Input : (N, 3, 88, 88)
    Output: (N, fc_units)
    """

    def __init__(self, fc_units: int = FC_UNITS) -> None:
        super().__init__()
        self.fc_units = fc_units
        flat_size = _cnn_output_size()

        self.conv_net = nn.Sequential(
            nn.Conv2d(CNN_CHANNELS[0], CNN_CHANNELS[1],
                      kernel_size=CNN_KERNELS[0], stride=CNN_STRIDES[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_CHANNELS[1], CNN_CHANNELS[2],
                      kernel_size=CNN_KERNELS[1], stride=CNN_STRIDES[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(CNN_CHANNELS[2], CNN_CHANNELS[3],
                      kernel_size=CNN_KERNELS[2], stride=CNN_STRIDES[2]),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_units),
            nn.ReLU(inplace=True),
        )
        self.apply(_orthogonal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, 3, 88, 88)  →  (N, fc_units)"""
        return self.fc(self.conv_net(x))


# ---------------------------------------------------------------------------
# Full Actor-Critic Agent
# ---------------------------------------------------------------------------
class MoltenpotAgent(nn.Module):
    """
    Generic recurrent Actor-Critic for MeltingPot substrates.

    Works with any discrete-action MeltingPot substrate by setting
    ``num_actions`` to match the substrate's action space size.

    Usage
    -----
    >>> agent = MoltenpotAgent(num_actions=9)          # Clean Up
    >>> agent = MoltenpotAgent(num_actions=7)          # Harvest / other
    >>> h = agent.initial_hidden(batch_size=4)         # (1, 4, gru_hidden)
    >>> logits, value, h = agent(obs, h)               # obs: (4, T, 3, 88, 88)
    """

    def __init__(
        self,
        num_actions: int = NUM_ACTIONS,
        fc_units: int = FC_UNITS,
        gru_hidden: int = GRU_HIDDEN,
        max_agents: int = MAX_AGENTS,
        num_scenarios: int = 0,
    ) -> None:
        super().__init__()
        self.num_actions   = num_actions
        self.fc_units      = fc_units
        self.gru_hidden    = gru_hidden
        self.max_agents    = max_agents
        # When >0, a one-hot scenario ID is appended to the CNN features (after
        # the agent one-hot) so the policy is told which scenario it is in and no
        # longer has to infer partner behaviour. 0 disables it (default), leaving
        # the architecture identical to the unconditioned model.
        self.num_scenarios = num_scenarios

        self.backbone = CNNBackbone(fc_units=fc_units)
        self.gru = nn.GRU(
            # CNN features + agent one-hot + (optional) scenario one-hot
            input_size=fc_units + max_agents + num_scenarios,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.actor  = nn.Linear(gru_hidden, num_actions)
        self.critic = nn.Linear(gru_hidden, 1)

        # Twin Q-networks for offline RL (BCQ, IQL, CQL)
        self.q1_head = nn.Linear(gru_hidden, num_actions)
        self.q2_head = nn.Linear(gru_hidden, num_actions)

        _orthogonal_init(self.gru, gain=ORTH_GAIN)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
        nn.init.orthogonal_(self.q1_head.weight, gain=1.0)
        nn.init.zeros_(self.q1_head.bias)
        nn.init.orthogonal_(self.q2_head.weight, gain=1.0)
        nn.init.zeros_(self.q2_head.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _augment_features(
        self,
        features: torch.Tensor,
        agent_ids: Optional[torch.Tensor],
        scenario_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Concatenate the agent one-hot (always) and, when ``num_scenarios > 0``,
        the scenario one-hot to CNN features before the GRU.

        Parameters
        ----------
        features     : (B, T, fc_units)
        agent_ids    : (B,) long — agent index per batch element, or None → zeros
        scenario_ids : (B,) long — scenario index per batch element, or None → zeros
                       (only used when ``num_scenarios > 0``)

        Returns
        -------
        (B, T, fc_units + max_agents [+ num_scenarios])
        """
        B, T = features.shape[:2]
        parts = [features]

        if agent_ids is not None:
            agent_oh = F.one_hot(agent_ids.long(), num_classes=self.max_agents).float()
            agent_oh = agent_oh.to(features.device).unsqueeze(1).expand(-1, T, -1)
        else:
            agent_oh = torch.zeros(B, T, self.max_agents, device=features.device)
        parts.append(agent_oh)

        if self.num_scenarios > 0:
            if scenario_ids is not None:
                scen_oh = F.one_hot(scenario_ids.long(), num_classes=self.num_scenarios).float()
                scen_oh = scen_oh.to(features.device).unsqueeze(1).expand(-1, T, -1)
            else:
                scen_oh = torch.zeros(B, T, self.num_scenarios, device=features.device)
            parts.append(scen_oh)

        return torch.cat(parts, dim=-1)

    def forward(
        self,
        obs: torch.Tensor,
        h_state: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        scenario_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs          : (B, T, 3, 88, 88)  float32 — normalised pixels
        h_state      : (1, B, gru_hidden)  float32 — GRU hidden state
        agent_ids    : (B,) long — agent index per batch element, or None
        scenario_ids : (B,) long — scenario index per batch element, or None
                       (ignored unless the model was built with num_scenarios > 0)

        Returns
        -------
        logits  : (B, T, num_actions)
        value   : (B, T, 1)
        h_state : (1, B, gru_hidden)  — detach at fragment boundaries
        """
        B, T = obs.shape[:2]
        x = obs.reshape(B * T, *obs.shape[2:])
        features = self.backbone(x).reshape(B, T, self.fc_units)
        features = self._augment_features(features, agent_ids, scenario_ids)
        gru_out, h_new = self.gru(features, h_state)
        return self.actor(gru_out), self.critic(gru_out), h_new

    def get_q_v(
        self,
        obs: torch.Tensor,
        h_state: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        scenario_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q1, Q2, V, and action logits for offline RL training.

        Parameters
        ----------
        agent_ids    : (B,) long — agent index per batch element, or None
        scenario_ids : (B,) long — scenario index per batch element, or None
                       (ignored unless the model was built with num_scenarios > 0)

        Returns: (q1, q2, v, logits, h_state_new)
        q1, q2, logits: (B, T, num_actions)
        v             : (B, T, 1)
        """
        B, T = obs.shape[:2]
        x = obs.reshape(B * T, *obs.shape[2:])
        features = self.backbone(x).reshape(B, T, self.fc_units)
        features = self._augment_features(features, agent_ids, scenario_ids)
        gru_out, h_new = self.gru(features, h_state)
        return (
            self.q1_head(gru_out),
            self.q2_head(gru_out),
            self.critic(gru_out),
            self.actor(gru_out),
            h_new,
        )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        obs_single: torch.Tensor,
        h_state: torch.Tensor,
    ) -> Tuple[int, float, float, torch.Tensor]:
        """
        Sample one action for a single step.

        Parameters
        ----------
        obs_single : (3, 88, 88)  — single frame
        h_state    : (1, 1, gru_hidden)

        Returns
        -------
        action   : int
        log_prob : float
        value    : float
        h_state  : (1, 1, gru_hidden)
        """
        obs = obs_single.unsqueeze(0).unsqueeze(0)
        logits, value, h_new = self.forward(obs, h_state)
        dist   = Categorical(logits=logits[0, 0])
        action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(value[0, 0, 0].item()),
            h_new,
        )

    @torch.no_grad()
    def act_batch(
        self,
        obs_batch: torch.Tensor,
        h_state: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        scenario_ids: Optional[torch.Tensor] = None,
    ):
        """
        Sample actions for multiple agents in a single forward pass.

        Parameters
        ----------
        obs_batch    : (A, 3, 88, 88)
        h_state      : (1, A, gru_hidden)
        agent_ids    : (A,) long — index of each agent, or None
        scenario_ids : (A,) long — scenario index of each agent, or None
                       (ignored unless the model was built with num_scenarios > 0)

        Returns
        -------
        actions   : np.ndarray (A,)  int32
        log_probs : np.ndarray (A,)  float32
        values    : np.ndarray (A,)  float32
        h_state   : (1, A, gru_hidden)
        """
        import numpy as np
        obs = obs_batch.unsqueeze(1)                    # (A, 1, 3, 88, 88)
        logits, value, h_new = self.forward(obs, h_state, agent_ids=agent_ids, scenario_ids=scenario_ids)
        dist      = Categorical(logits=logits[:, 0])    # (A, num_actions)
        actions   = dist.sample()
        log_probs = dist.log_prob(actions)
        return (
            actions.cpu().numpy().astype(np.int32),
            log_probs.cpu().numpy().astype(np.float32),
            value[:, 0, 0].cpu().numpy().astype(np.float32),
            h_new,
        )

    def initial_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Zero-initialised GRU hidden state: (1, batch_size, gru_hidden)."""
        return torch.zeros(1, batch_size, self.gru_hidden)

    # ------------------------------------------------------------------
    # Weight serialisation (Ray weight sharing)
    # ------------------------------------------------------------------
    def get_weights(self) -> dict:
        """Return CPU state dict for transfer across Ray workers."""
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict) -> None:
        """Load a state dict received from the learner."""
        self.load_state_dict(weights)
