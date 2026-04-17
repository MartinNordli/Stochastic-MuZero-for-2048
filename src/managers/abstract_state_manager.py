"""Abstract State Manager (ASM).

Wraps the five Stochastic-MuZero networks with jit'd forwards so the rest of
the code (u-MCTS, training) talks to one clean interface. Does *not* hold
parameters — parameters live in the NeuralNetworkManager's TrainState. The
ASM takes a TrainState at construction or per-call via `set_params`.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


class AbstractStateManager:
    def __init__(self, nets, num_chance_codes: int, num_actions: int):
        self.nets = nets
        self.num_chance_codes = num_chance_codes
        self.num_actions = num_actions
        self.params = None

        # Pre-jit the five forwards. `params` is captured via an explicit arg.
        self._f_repr = jax.jit(nets.apply_repr)
        self._f_phi_core = jax.jit(nets.apply_phi)
        self._f_psi = jax.jit(nets.apply_psi)
        self._f_dyn_core = jax.jit(nets.apply_dyn)
        self._f_pred = jax.jit(nets.apply_pred)

    def set_params(self, params) -> None:
        self.params = params

    # --- forwards used during search (single example, no batch axis) --------

    def repr(self, board: np.ndarray) -> np.ndarray:
        """Encoded board -> abstract state."""
        return np.asarray(self._f_repr(self.params, jnp.asarray(board, dtype=jnp.float32)))

    def phi(self, s: np.ndarray, action: int) -> np.ndarray:
        a = jnp.asarray(np.eye(self.num_actions, dtype=np.float32)[action])
        return np.asarray(self._f_phi_core(self.params, jnp.asarray(s), a))

    def psi_logits(self, sa: np.ndarray):
        q, sigma_logits = self._f_psi(self.params, jnp.asarray(sa))
        return float(q), np.asarray(sigma_logits)

    def dynamics(self, sa: np.ndarray, code: int):
        c = jnp.asarray(np.eye(self.num_chance_codes, dtype=np.float32)[code])
        s_next, r = self._f_dyn_core(self.params, jnp.asarray(sa), c)
        return np.asarray(s_next), float(r)

    def predict_logits(self, s: np.ndarray):
        pi_logits, v = self._f_pred(self.params, jnp.asarray(s))
        return np.asarray(pi_logits), float(v)

    def predict(self, s: np.ndarray):
        pi_logits, v = self.predict_logits(s)
        pi = _softmax(pi_logits)
        return pi, v


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()
