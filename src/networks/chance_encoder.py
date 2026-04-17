"""Chance encoder (training-only).

Stochastic MuZero uses a fixed-size latent alphabet of K chance codes. At
training time, we need *targets* for the chance distribution sigma(s^a) and
the discrete code to feed into NN_d. The encoder takes the real (abstract)
next state and produces a code. We train it with a Gumbel-softmax
straight-through estimator so the discrete draw remains differentiable.

Because we keep the alphabet abstract, we do NOT need to enumerate all real
2048 spawns — the network discovers its own partition of chance space.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class ChanceEncoder(nn.Module):
    """Maps the real post-spawn abstract state to a one-hot-ish code.

    Input: concatenation of (s^a, s_next) where both come from the
    representation network. Output: code_logits of shape (K,). A
    Gumbel-softmax draw yields a differentiable approximation to a one-hot.
    """
    num_chance_codes: int
    hidden: int = 128

    @nn.compact
    def __call__(self, sa: jnp.ndarray, s_next: jnp.ndarray) -> jnp.ndarray:
        h = jnp.concatenate([sa, s_next], axis=-1)
        h = nn.Dense(self.hidden)(h)
        h = nn.gelu(h)
        logits = nn.Dense(self.num_chance_codes)(h)
        return logits


def gumbel_softmax(
    rng: jax.Array, logits: jnp.ndarray, temperature: float, hard: bool
) -> jnp.ndarray:
    """Straight-through Gumbel-softmax (forward is one-hot, backward is soft)."""
    g = -jnp.log(-jnp.log(jax.random.uniform(rng, logits.shape) + 1e-20) + 1e-20)
    y_soft = jax.nn.softmax((logits + g) / temperature, axis=-1)
    if not hard:
        return y_soft
    idx = jnp.argmax(y_soft, axis=-1)
    y_hard = jax.nn.one_hot(idx, logits.shape[-1])
    return y_hard + jax.lax.stop_gradient(y_hard - y_soft) * 0 + (y_hard - y_soft)  # ST


def straight_through_onehot(y_soft: jnp.ndarray) -> jnp.ndarray:
    """Cleanest ST: forward uses argmax one-hot, backward uses y_soft grads."""
    idx = jnp.argmax(y_soft, axis=-1)
    y_hard = jax.nn.one_hot(idx, y_soft.shape[-1])
    return y_soft + jax.lax.stop_gradient(y_hard - y_soft)
