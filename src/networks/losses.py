"""Loss functions for Stochastic MuZero BPTT training.

Everything is a pure function of (predictions, targets) so train_step can
wrap them in a single value_and_grad call and Optax will handle the rest.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def softmax_cross_entropy(logits: jnp.ndarray, target_probs: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(target_probs * log_probs, axis=-1)


def squared_error(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return (pred - target) ** 2


def l2_regularisation(params: dict) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(params)
    return sum(jnp.sum(p ** 2) for p in leaves)
