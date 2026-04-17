"""The five neural networks of Stochastic MuZero.

Notation:
    s       = abstract (latent) state after spawn,        shape (abstract_dim,)
    s^a     = abstract afterstate (after agent action),   shape (abstract_dim,)
    c       = categorical chance code, one-hot over K     shape (K,)
    a       = categorical action,     one-hot over |A|    shape (num_actions,)

    NN_r : board_stack        -> s              (representation)
    NN_phi: (s, a_onehot)     -> s^a            (afterstate dynamics)
    NN_psi: s^a               -> (Q, sigma)     (afterstate prediction: value + chance distribution)
    NN_d  : (s^a, c_onehot)   -> (s', r)        (dynamics)
    NN_p  : s                 -> (pi, v)        (prediction)

A chance encoder lives in chance_encoder.py and is used only during training.

All five networks are Flax linen modules with pure, jit-friendly forward
passes. Parameter trees live in a single composite TrainState.
"""
from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class RepresentationNet(nn.Module):
    """NN_r : (H, W, C_in) or (stack, H, W, C_in) -> abstract state."""
    repr_channels: int
    abstract_dim: int

    @nn.compact
    def __call__(self, board_stack: jnp.ndarray) -> jnp.ndarray:
        x = board_stack
        # If a history stack was passed, concatenate along the channel axis.
        if x.ndim == 4:  # (stack, H, W, C)
            x = jnp.transpose(x, (1, 2, 0, 3))
            x = x.reshape(x.shape[0], x.shape[1], -1)
        x = nn.Conv(self.repr_channels, (3, 3), padding="SAME")(x)
        x = nn.gelu(x)
        x = nn.Conv(self.repr_channels, (3, 3), padding="SAME")(x)
        x = nn.gelu(x)
        x = x.reshape(-1)
        x = nn.Dense(self.abstract_dim)(x)
        x = nn.LayerNorm()(x)
        return x


class AfterstateDynamicsNet(nn.Module):
    """NN_phi: (s, a_onehot) -> s^a."""
    abstract_dim: int
    hidden: int

    @nn.compact
    def __call__(self, s: jnp.ndarray, a_onehot: jnp.ndarray) -> jnp.ndarray:
        h = jnp.concatenate([s, a_onehot], axis=-1)
        h = nn.Dense(self.hidden)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden)(h)
        h = nn.gelu(h)
        delta = nn.Dense(self.abstract_dim)(h)
        return nn.LayerNorm()(s + delta)


class AfterstatePredictionNet(nn.Module):
    """NN_psi: s^a -> (Q_scalar, sigma_logits[K])."""
    hidden: int
    num_chance_codes: int

    @nn.compact
    def __call__(self, sa: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = nn.Dense(self.hidden)(sa)
        h = nn.gelu(h)
        q = nn.Dense(1)(h).squeeze(-1)
        sigma_logits = nn.Dense(self.num_chance_codes)(h)
        return q, sigma_logits


class DynamicsNet(nn.Module):
    """NN_d: (s^a, c_onehot) -> (s', reward_scalar)."""
    abstract_dim: int
    hidden: int

    @nn.compact
    def __call__(
        self, sa: jnp.ndarray, c_onehot: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = jnp.concatenate([sa, c_onehot], axis=-1)
        h = nn.Dense(self.hidden)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden)(h)
        h = nn.gelu(h)
        delta = nn.Dense(self.abstract_dim)(h)
        s_next = nn.LayerNorm()(sa + delta)
        reward = nn.Dense(1)(h).squeeze(-1)
        return s_next, reward


class PredictionNet(nn.Module):
    """NN_p: s -> (pi_logits[|A|], v_scalar)."""
    hidden: int
    num_actions: int

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = nn.Dense(self.hidden)(s)
        h = nn.gelu(h)
        pi_logits = nn.Dense(self.num_actions)(h)
        v = nn.Dense(1)(h).squeeze(-1)
        return pi_logits, v


class MuZeroNets:
    """Container that holds unbound Flax modules + initialised parameter tree.

    Not a Flax module itself; acts as a clean factory / forward wrapper so the
    rest of the codebase never imports flax directly.
    """
    def __init__(
        self,
        num_actions: int,
        num_chance_codes: int,
        abstract_dim: int,
        repr_channels: int,
        afterstate_hidden: int,
        dynamics_hidden: int,
        prediction_hidden: int,
    ):
        self.num_actions = num_actions
        self.num_chance_codes = num_chance_codes
        self.abstract_dim = abstract_dim
        self.repr = RepresentationNet(repr_channels, abstract_dim)
        self.phi = AfterstateDynamicsNet(abstract_dim, afterstate_hidden)
        self.psi = AfterstatePredictionNet(afterstate_hidden, num_chance_codes)
        self.dyn = DynamicsNet(abstract_dim, dynamics_hidden)
        self.pred = PredictionNet(prediction_hidden, num_actions)

    def init(self, rng: jax.Array, board_shape: Tuple[int, int, int]) -> dict:
        """Initialise parameters for all five networks. Returns a single dict."""
        r_rng, phi_rng, psi_rng, d_rng, p_rng = jax.random.split(rng, 5)
        dummy_board = jnp.zeros(board_shape, dtype=jnp.float32)
        dummy_s = jnp.zeros((self.abstract_dim,), dtype=jnp.float32)
        dummy_a = jnp.zeros((self.num_actions,), dtype=jnp.float32)
        dummy_c = jnp.zeros((self.num_chance_codes,), dtype=jnp.float32)
        return {
            "repr": self.repr.init(r_rng, dummy_board),
            "phi":  self.phi.init(phi_rng, dummy_s, dummy_a),
            "psi":  self.psi.init(psi_rng, dummy_s),
            "dyn":  self.dyn.init(d_rng, dummy_s, dummy_c),
            "pred": self.pred.init(p_rng, dummy_s),
        }

    # ------- forward helpers (pure; jit'd by callers that keep them hot) -------

    def apply_repr(self, params, board):
        return self.repr.apply(params["repr"], board)

    def apply_phi(self, params, s, a_onehot):
        return self.phi.apply(params["phi"], s, a_onehot)

    def apply_psi(self, params, sa):
        return self.psi.apply(params["psi"], sa)

    def apply_dyn(self, params, sa, c_onehot):
        return self.dyn.apply(params["dyn"], sa, c_onehot)

    def apply_pred(self, params, s):
        return self.pred.apply(params["pred"], s)
