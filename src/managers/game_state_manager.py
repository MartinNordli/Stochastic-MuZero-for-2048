"""Game State Manager (GSM).

A thin adapter that gives the rest of the system a domain-agnostic handle on
the current game state. Wraps a SimWorld (e.g. Game2048) and tracks the live
state + an RNG for the stochastic transition.
"""
from __future__ import annotations

import numpy as np

from src.games.game_base import SimWorld


class GameStateManager:
    def __init__(self, world: SimWorld, rng: np.random.Generator):
        self.world = world
        self.rng = rng
        self.state: np.ndarray = world.initial_state(rng)

    def reset(self) -> np.ndarray:
        self.state = self.world.initial_state(self.rng)
        return self.state

    def step(self, action: int):
        next_state, reward, done = self.world.step(self.state, action, self.rng)
        self.state = next_state
        return next_state, reward, done

    def legal_actions(self) -> np.ndarray:
        return self.world.legal_actions(self.state)

    def is_terminal(self) -> bool:
        return self.world.is_terminal(self.state)

    def encode(self, state: np.ndarray | None = None) -> np.ndarray:
        return self.world.encode(self.state if state is None else state)
