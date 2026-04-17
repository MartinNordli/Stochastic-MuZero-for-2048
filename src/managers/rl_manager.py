"""Reinforcement Learning Manager (RLM).

Runs episodes:
    * reset the game via the GSM
    * for each step: encode the lookback, compute abstract root state via NN_r,
      run u-MCTS to get (pi, v*), sample an action with the current
      temperature, step the simulator, record the (s, a, r, pi, v*) tuple
    * at episode end, build an Episode and push it to the EpisodeBuffer

Also exposes `play_actor`, which deploys the trained agent WITHOUT u-MCTS —
policy comes straight from NN_r + NN_p, per the project spec.
"""
from __future__ import annotations

from typing import List

import numpy as np

from src.mcts.umcts import MCTSConfig, StochasticUMCTS
from src.managers.abstract_state_manager import AbstractStateManager
from src.managers.game_state_manager import GameStateManager
from src.training.episode_buffer import Episode


def _sample_action(pi: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 1e-6:
        return int(np.argmax(pi))
    logits = np.log(pi + 1e-12) / temperature
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


class RLManager:
    def __init__(
        self,
        gsm: GameStateManager,
        asm: AbstractStateManager,
        mcts_cfg: MCTSConfig,
        max_steps: int,
        q: int,
        rng: np.random.Generator,
    ):
        self.gsm = gsm
        self.asm = asm
        self.mcts_cfg = mcts_cfg
        self.max_steps = max_steps
        self.q = q
        self.rng = rng

    # ---- self-play: one episode with u-MCTS ---------------------------------

    def run_episode(self, temperature: float) -> Episode:
        self.gsm.reset()
        states: List[np.ndarray] = [self.gsm.state.copy()]
        actions: List[int] = []
        rewards: List[float] = []
        policies: List[np.ndarray] = []
        values: List[float] = []

        mcts = StochasticUMCTS(self.asm, self.mcts_cfg, self.rng)

        for _ in range(self.max_steps):
            if self.gsm.is_terminal():
                break
            legal = self.gsm.legal_actions()
            if not legal.any():
                break

            obs_window = self._lookback(states)   # (q+1, H, W)
            obs_onehot = self._encode_window(obs_window)
            root_s = self.asm.repr(obs_onehot)

            pi, v, _root = mcts.run(root_s, legal)
            # Zero-out illegal actions (defense in depth; MCTS already filters).
            pi = pi * legal.astype(pi.dtype)
            s = pi.sum()
            pi = pi / s if s > 0 else legal.astype(pi.dtype) / legal.sum()

            action = _sample_action(pi, temperature, self.rng)
            _next_state, reward, _done = self.gsm.step(action)

            actions.append(action)
            rewards.append(float(reward))
            policies.append(pi)
            values.append(float(v))
            states.append(self.gsm.state.copy())

        return Episode(
            states=np.stack(states, axis=0).astype(np.int8),
            actions=np.asarray(actions, dtype=np.int32),
            rewards=np.asarray(rewards, dtype=np.float32),
            policies=np.stack(policies, axis=0) if policies else np.zeros((0, self.mcts_cfg.num_actions), np.float32),
            values=np.asarray(values, dtype=np.float32),
        )

    # ---- deploy-time: no MCTS, just NN_r + NN_p ---------------------------

    def play_actor(self, temperature: float = 0.0):
        """Generator yielding (state, pi, action, reward) tuples until the game ends.

        Uses only the representation + prediction networks — the trained actor.
        """
        self.gsm.reset()
        states = [self.gsm.state.copy()]
        while not self.gsm.is_terminal():
            legal = self.gsm.legal_actions()
            if not legal.any():
                break
            obs_window = self._lookback(states)
            obs_onehot = self._encode_window(obs_window)
            s = self.asm.repr(obs_onehot)
            pi, _v = self.asm.predict(s)
            pi = pi * legal.astype(pi.dtype)
            total = pi.sum()
            pi = pi / total if total > 0 else legal.astype(pi.dtype) / legal.sum()
            action = _sample_action(pi, temperature, self.rng)
            _next_state, reward, _done = self.gsm.step(action)
            yield states[-1], pi, action, float(reward), self.gsm.state.copy()
            states.append(self.gsm.state.copy())

    # ---- internals --------------------------------------------------------

    def _lookback(self, states: List[np.ndarray]) -> np.ndarray:
        window = np.zeros((self.q + 1, *states[-1].shape), dtype=np.int8)
        for j, idx in enumerate(range(len(states) - 1 - self.q, len(states))):
            if 0 <= idx < len(states):
                window[j] = states[idx]
        return window

    def _encode_window(self, window: np.ndarray) -> np.ndarray:
        max_log2 = self.gsm.world.max_log2
        out = np.zeros((*window.shape, max_log2 + 1), dtype=np.float32)
        clipped = np.clip(window, 0, max_log2)
        for i in range(window.shape[0]):
            rows, cols = np.indices(window.shape[1:])
            out[i, rows, cols, clipped[i]] = 1.0
        return out
