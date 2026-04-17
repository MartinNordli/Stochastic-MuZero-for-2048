"""Neural Network Manager (NNM).

Owns parameters for the five networks + the chance encoder, and defines the
jit-compiled BPTT training step that unrolls w steps and sums policy / value /
reward / chance losses.

The unroll, per sample (shapes for one sample, no batch axis):

    obs_window       (q+1, H, W)       int8   — look-back board history
    actions          (w,)              int32
    target_policy    (w+1, A)          float32
    target_value     (w+1,)            float32
    target_reward    (w,)              float32
    next_obs         (w, H, W)         int8   — real next boards (for chance targets)
    mask             (w+1,)            float32

Procedure:
    s_0           = NN_r(onehot(obs_window))
    pi_0, v_0     = NN_p(s_0)
    for j in 0..w-1:
        sa_j       = NN_phi(s_j, onehot(actions_j))
        Q_j, sig_j = NN_psi(sa_j)
        c_j        = chance_encoder(sa_j, NN_r(onehot(next_obs[j])))     # soft Gumbel + ST
        s_{j+1}, r_j = NN_d(sa_j, c_j)
        pi_{j+1}, v_{j+1} = NN_p(s_{j+1})

Losses summed over j:
    policy  = Σ mask[j] · CE(target_policy[j], pi_logits_j)
    value   = Σ mask[j] · MSE(target_value[j], v_j)
    reward  = Σ mask[j+1] · MSE(target_reward[j], r_j)       (for j in 0..w-1)
    chance  = Σ mask[j+1] · CE(stop_grad(c_j), sig_j_logits) (soft-target CE)
    afterstate = Σ mask[j+1] · MSE(Q_j, target_value[j+1])    (optional aux)
    L2     = Σ params²

This is a faithful BPTT through the composite graph; JAX traces it end-to-end
and returns gradients for every parameter in a single value_and_grad call.
"""
from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.networks.chance_encoder import ChanceEncoder, straight_through_onehot, gumbel_softmax
from src.networks.losses import l2_regularisation, softmax_cross_entropy, squared_error
from src.networks.networks import MuZeroNets


class NeuralNetworkManager:
    def __init__(
        self,
        nets: MuZeroNets,
        grid_size: int,
        max_log2: int,
        w: int,
        q: int,
        lr: float,
        l2: float,
        gumbel_temperature: float,
        num_actions: int,
        num_chance_codes: int,
    ):
        self.nets = nets
        self.grid_size = grid_size
        self.max_log2 = max_log2
        self.max_log2_plus_one = max_log2 + 1
        self.w = w
        self.q = q
        self.l2 = l2
        self.gumbel_temperature = gumbel_temperature
        self.num_actions = num_actions
        self.num_chance_codes = num_chance_codes

        self.chance_encoder = ChanceEncoder(num_chance_codes=num_chance_codes)

        self.optimizer = optax.adam(lr)

        # Lazily jit train_step with static shapes after init.
        self._train_step = jax.jit(self._train_step_impl)

    # ---- init ---------------------------------------------------------------

    def init_params(self, rng: jax.Array):
        r_key, enc_key = jax.random.split(rng)
        params = self.nets.init(
            r_key,
            board_shape=(self.grid_size, self.grid_size, self.max_log2_plus_one),
        )
        # Chance encoder needs (sa, s_next) dummy inputs to initialise.
        dummy_sa = jnp.zeros((self.nets.abstract_dim,), dtype=jnp.float32)
        dummy_sn = jnp.zeros((self.nets.abstract_dim,), dtype=jnp.float32)
        enc_params = self.chance_encoder.init(enc_key, dummy_sa, dummy_sn)
        opt_state = self.optimizer.init((params, enc_params))
        return params, enc_params, opt_state

    # ---- a single-sample unroll (used under vmap in train_step) -------------

    def _unroll_one(self, params, enc_params, sample, rng):
        obs_window = sample["obs"]         # (q+1, H, W)
        actions = sample["actions"]        # (w,)
        target_policy = sample["target_policy"]  # (w+1, A)
        target_value = sample["target_value"]    # (w+1,)
        target_reward = sample["target_reward"]  # (w,)
        next_obs = sample["next_obs"]            # (w, H, W)
        mask = sample["mask"]                    # (w+1,)

        def one_hot_board(board):
            clipped = jnp.clip(board.astype(jnp.int32), 0, self.max_log2)
            return jax.nn.one_hot(clipped, self.max_log2_plus_one)

        obs_onehot = one_hot_board(obs_window)            # (q+1, H, W, C)
        s = self.nets.apply_repr(params, obs_onehot)

        pi_logits, v = self.nets.apply_pred(params, s)

        policy_loss = mask[0] * softmax_cross_entropy(pi_logits, target_policy[0])
        value_loss = mask[0] * squared_error(v, target_value[0])
        reward_loss = 0.0
        chance_loss = 0.0
        afterstate_loss = 0.0

        for j in range(self.w):
            a_oh = jax.nn.one_hot(actions[j], self.num_actions)
            sa = self.nets.apply_phi(params, s, a_oh)
            q_af, sigma_logits = self.nets.apply_psi(params, sa)

            # chance encoder gets NN_r on the *real* next board.
            next_onehot = one_hot_board(next_obs[j])       # (H, W, C)
            s_real_next = self.nets.apply_repr(params, next_onehot)
            code_logits = self.chance_encoder.apply(enc_params, sa, s_real_next)

            rng, sub = jax.random.split(rng)
            code_soft = gumbel_softmax(sub, code_logits, self.gumbel_temperature, hard=False)
            code_onehot = straight_through_onehot(code_soft)

            s, r_pred = self.nets.apply_dyn(params, sa, code_onehot)
            pi_logits, v = self.nets.apply_pred(params, s)

            m = mask[j + 1]
            reward_loss += m * squared_error(r_pred, target_reward[j])
            chance_loss += m * softmax_cross_entropy(
                sigma_logits, jax.lax.stop_gradient(code_onehot)
            )
            afterstate_loss += m * squared_error(q_af, target_value[j + 1])
            policy_loss += m * softmax_cross_entropy(pi_logits, target_policy[j + 1])
            value_loss += m * squared_error(v, target_value[j + 1])

        total = policy_loss + value_loss + reward_loss + chance_loss + 0.25 * afterstate_loss
        return {
            "total": total,
            "policy": policy_loss,
            "value": value_loss,
            "reward": reward_loss,
            "chance": chance_loss,
            "afterstate": afterstate_loss,
        }

    # ---- train step (jit'd)  ------------------------------------------------

    def _train_step_impl(self, params, enc_params, opt_state, batch, rng):
        mbs = batch["actions"].shape[0]
        keys = jax.random.split(rng, mbs)

        def loss_fn(params, enc_params):
            def one_sample_fn(obs, actions, tp, tv, tr, nobs, mask, key):
                sample = {
                    "obs": obs, "actions": actions,
                    "target_policy": tp, "target_value": tv,
                    "target_reward": tr, "next_obs": nobs, "mask": mask,
                }
                return self._unroll_one(params, enc_params, sample, key)

            losses = jax.vmap(one_sample_fn)(
                batch["obs"], batch["actions"],
                batch["target_policy"], batch["target_value"],
                batch["target_reward"], batch["next_obs"],
                batch["mask"], keys,
            )
            mean_losses = {k: jnp.mean(v) for k, v in losses.items()}
            l2_penalty = l2_regularisation(params) + l2_regularisation(enc_params)
            mean_losses["total"] = mean_losses["total"] + self.l2 * l2_penalty
            mean_losses["l2"] = l2_penalty
            return mean_losses["total"], mean_losses

        (_, loss_dict), grads = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(
            params, enc_params,
        )
        updates, opt_state = self.optimizer.update(grads, opt_state)
        new_params, new_enc_params = optax.apply_updates((params, enc_params), updates)
        return new_params, new_enc_params, opt_state, loss_dict

    def train_step(self, params, enc_params, opt_state, batch, rng):
        # Convert numpy arrays to JAX arrays once (jit caches the traced fn).
        jax_batch = {k: jnp.asarray(v) for k, v in batch.items()}
        return self._train_step(params, enc_params, opt_state, jax_batch, rng)

    # ---- checkpointing ------------------------------------------------------

    def save(self, path: str | Path, params, enc_params, opt_state, meta: dict):
        payload = {
            "params": flax.serialization.to_state_dict(params),
            "enc_params": flax.serialization.to_state_dict(enc_params),
            "opt_state": flax.serialization.to_state_dict(opt_state),
            "meta": meta,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str | Path, params_like, enc_params_like, opt_state_like):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        params = flax.serialization.from_state_dict(params_like, payload["params"])
        enc_params = flax.serialization.from_state_dict(enc_params_like, payload["enc_params"])
        opt_state = flax.serialization.from_state_dict(opt_state_like, payload["opt_state"])
        return params, enc_params, opt_state, payload.get("meta", {})
