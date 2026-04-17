"""Episode buffer (EB).

Stores per-episode trajectories produced by the RLM. Each step has:
    state_k      — raw game state (needed to re-encode on the fly)
    action_k     — action chosen at step k
    reward_k     — reward received transitioning to state_{k+1}
    policy_k     — u-MCTS visit distribution at step k (pi*)
    value_k      — u-MCTS root value at step k (v*)
    done_k       — whether state_k is terminal (useful for padding truncation)

For Stochastic MuZero BPTT we also need access to the *next* raw state at each
step to derive chance-code targets via the chance encoder. We keep states[k+1]
around naturally (it's just states[k+1] in the sequence).

Sampling returns batched tensors of length (q+1) for the lookback and (w) for
the unroll, with zero-padding at episode boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Episode:
    states: np.ndarray        # (T+1, H, W) int8 — includes the terminal state
    actions: np.ndarray       # (T,) int
    rewards: np.ndarray       # (T,) float32
    policies: np.ndarray      # (T, num_actions) float32
    values: np.ndarray        # (T,) float32

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class EpisodeBuffer:
    capacity: int = 2000
    _episodes: List[Episode] = field(default_factory=list)

    def add(self, episode: Episode) -> None:
        self._episodes.append(episode)
        if len(self._episodes) > self.capacity:
            self._episodes.pop(0)

    def __len__(self) -> int:
        return len(self._episodes)

    def total_steps(self) -> int:
        return sum(len(e) for e in self._episodes)

    # ---- sampling ----

    def sample_batch(
        self,
        rng: np.random.Generator,
        mbs: int,
        q: int,
        w: int,
        num_actions: int,
    ):
        """Return a dict of ndarrays with leading axis `mbs`.

        Keys:
            obs           — (mbs, q+1, H, W)  int8   (look-back window)
            actions       — (mbs, w)          int32
            target_policy — (mbs, w+1, A)     float32
            target_value  — (mbs, w+1)        float32
            target_reward — (mbs, w)          float32
            next_obs      — (mbs, w, H, W)    int8   (used for chance targets)
            mask          — (mbs, w+1)        float32 (1 for valid, 0 for padded)
        """
        if not self._episodes:
            raise RuntimeError("buffer is empty")

        obs_list, actions_list = [], []
        pi_list, v_list, r_list, next_obs_list, mask_list = [], [], [], [], []

        H, W = self._episodes[0].states.shape[1:]

        for _ in range(mbs):
            ep = self._episodes[rng.integers(0, len(self._episodes))]
            T = len(ep)
            k = int(rng.integers(0, T))   # sample a step in [0, T-1]

            # ---- lookback obs: states[k-q .. k] (pad with zeros) ----
            obs = np.zeros((q + 1, H, W), dtype=np.int8)
            for j, idx in enumerate(range(k - q, k + 1)):
                if 0 <= idx < ep.states.shape[0]:
                    obs[j] = ep.states[idx]
            # ---- roll-ahead targets over w steps ----
            actions = np.zeros(w, dtype=np.int32)
            pi = np.zeros((w + 1, num_actions), dtype=np.float32)
            v = np.zeros(w + 1, dtype=np.float32)
            r = np.zeros(w, dtype=np.float32)
            next_obs = np.zeros((w, H, W), dtype=np.int8)
            mask = np.zeros(w + 1, dtype=np.float32)

            # j=0 target pulled at step k
            if k < T:
                pi[0] = ep.policies[k]
                v[0] = ep.values[k]
                mask[0] = 1.0

            for j in range(w):
                step = k + j
                if step < T:
                    actions[j] = ep.actions[step]
                    r[j] = ep.rewards[step]
                    next_obs[j] = ep.states[step + 1] if step + 1 < ep.states.shape[0] else 0
                    # target pi/value at step+1
                    if step + 1 < T:
                        pi[j + 1] = ep.policies[step + 1]
                        v[j + 1] = ep.values[step + 1]
                        mask[j + 1] = 1.0
                    else:
                        # terminal — zero targets, zero mask (or use a 0 value for bootstrap)
                        pi[j + 1] = np.ones(num_actions, dtype=np.float32) / num_actions
                        v[j + 1] = 0.0
                        mask[j + 1] = 1.0  # still count terminal step
                else:
                    # past episode end — leave zeros, mask off
                    pass

            obs_list.append(obs)
            actions_list.append(actions)
            pi_list.append(pi)
            v_list.append(v)
            r_list.append(r)
            next_obs_list.append(next_obs)
            mask_list.append(mask)

        return {
            "obs":           np.stack(obs_list, axis=0),
            "actions":       np.stack(actions_list, axis=0),
            "target_policy": np.stack(pi_list, axis=0),
            "target_value":  np.stack(v_list, axis=0),
            "target_reward": np.stack(r_list, axis=0),
            "next_obs":      np.stack(next_obs_list, axis=0),
            "mask":          np.stack(mask_list, axis=0),
        }
