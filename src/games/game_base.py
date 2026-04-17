"""Abstract SimWorld interface.

Any new game (e.g. BitFall, 2048, a maze) implements this interface and can
be dropped into the MuZero pipeline with no changes to the AI side.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class SimWorld(ABC):
    """Contract for a stochastic, single-player environment.

    The environment transition is split into a deterministic 'move' (agent
    action) and a stochastic 'spawn' (environment outcome). This split is
    what Stochastic MuZero's u-MCTS walks over via afterstate + chance nodes.
    """

    num_actions: int
    state_shape: Tuple[int, ...]
    max_log2: int  # size of one-hot channel dimension for encoding

    @abstractmethod
    def initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Return a fresh initial game state."""

    @abstractmethod
    def legal_actions(self, state: np.ndarray) -> np.ndarray:
        """Return a bool mask of shape (num_actions,)."""

    @abstractmethod
    def move(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
        """Deterministic part: apply agent action. Return (afterstate, reward, changed)."""

    @abstractmethod
    def spawn(self, afterstate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Stochastic part: environment draws a random outcome from the afterstate."""

    @abstractmethod
    def step(
        self, state: np.ndarray, action: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, float, bool]:
        """Convenience: move + spawn in one call. Returns (next_state, reward, done)."""

    @abstractmethod
    def is_terminal(self, state: np.ndarray) -> bool:
        ...

    @abstractmethod
    def encode(self, state: np.ndarray) -> np.ndarray:
        """Map a raw state to the one-hot tensor the representation network expects."""
