"""2048 SimWorld.

State representation: np.int8[grid_size, grid_size] of log2(tile_value), with 0
for empty cells. E.g. a cell holding the tile '8' stores the value 3.

The transition is split into two parts:

    move(state, a)      -> (afterstate, reward, changed)     # deterministic
    spawn(afterstate)   -> next_state                        # stochastic

step(state, a) chains them. Stochastic MuZero's u-MCTS walks the split
explicitly — decision nodes branch on actions, chance nodes branch on spawns.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .game_base import SimWorld

# Action encoding.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class Game2048(SimWorld):
    num_actions = 4

    def __init__(self, grid_size: int = 4, spawn_two_prob: float = 0.9, max_log2: int = 15):
        self.grid_size = grid_size
        self.spawn_two_prob = spawn_two_prob
        # +1 because channel 0 is "empty".
        self.max_log2 = max_log2
        self.state_shape = (grid_size, grid_size)

    # ---------- core slide/merge (operates on a single row, left-ward) ----------

    @staticmethod
    def _slide_row_left(row: np.ndarray) -> Tuple[np.ndarray, int]:
        """Slide a 1-D row to the left, merging equal neighbours once.

        `row` holds log2 values (0 = empty). Returns (new_row, reward) where
        reward is the sum of *tile values* (2**log2) produced by merges —
        the classic 2048 score delta.
        """
        n = row.shape[0]
        out = np.zeros_like(row)
        reward = 0
        write = 0
        last = 0  # 0 means "no pending tile"
        for i in range(n):
            v = row[i]
            if v == 0:
                continue
            if last == 0:
                last = v
            elif last == v:
                merged = last + 1  # log2 increment
                out[write] = merged
                reward += 1 << merged  # i.e. 2**merged
                write += 1
                last = 0
            else:
                out[write] = last
                write += 1
                last = v
        if last != 0:
            out[write] = last
        return out, reward

    # ---------- SimWorld interface ----------

    def initial_state(self, rng: np.random.Generator) -> np.ndarray:
        state = np.zeros(self.state_shape, dtype=np.int8)
        state = self.spawn(state, rng)
        state = self.spawn(state, rng)
        return state

    def move(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
        # Rotate the board so that `action` becomes "slide left", slide, rotate
        # back. UP = slide up, which is equivalent to transposing, sliding
        # columns left, transposing back.
        board = state.copy()
        if action == UP:
            board = board.T
        elif action == DOWN:
            board = board.T[:, ::-1]
        elif action == LEFT:
            pass
        elif action == RIGHT:
            board = board[:, ::-1]
        else:
            raise ValueError(f"unknown action {action}")

        total_reward = 0
        new_rows = np.empty_like(board)
        for r in range(board.shape[0]):
            new_rows[r], row_reward = self._slide_row_left(board[r])
            total_reward += row_reward

        if action == UP:
            new_rows = new_rows.T
        elif action == DOWN:
            new_rows = new_rows[:, ::-1].T
        elif action == RIGHT:
            new_rows = new_rows[:, ::-1]

        changed = not np.array_equal(new_rows, state)
        return new_rows, float(total_reward), changed

    def spawn(self, afterstate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        empties = np.argwhere(afterstate == 0)
        if empties.size == 0:
            return afterstate
        idx = empties[rng.integers(0, len(empties))]
        value = 1 if rng.random() < self.spawn_two_prob else 2  # log2 value
        new_state = afterstate.copy()
        new_state[tuple(idx)] = value
        return new_state

    def step(
        self, state: np.ndarray, action: int, rng: np.random.Generator
    ) -> Tuple[np.ndarray, float, bool]:
        afterstate, reward, changed = self.move(state, action)
        if not changed:
            # Illegal action in 2048: board doesn't change, no tile spawned, no reward.
            return state, 0.0, self.is_terminal(state)
        next_state = self.spawn(afterstate, rng)
        return next_state, reward, self.is_terminal(next_state)

    def legal_actions(self, state: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=bool)
        for a in range(self.num_actions):
            _, _, changed = self.move(state, a)
            mask[a] = changed
        return mask

    def is_terminal(self, state: np.ndarray) -> bool:
        return not self.legal_actions(state).any()

    def encode(self, state: np.ndarray) -> np.ndarray:
        """One-hot encode along the tile-value axis.

        Output shape: (grid_size, grid_size, max_log2 + 1), float32.
        """
        clipped = np.clip(state, 0, self.max_log2).astype(np.int64)
        onehot = np.zeros((*self.state_shape, self.max_log2 + 1), dtype=np.float32)
        rows, cols = np.indices(self.state_shape)
        onehot[rows, cols, clipped] = 1.0
        return onehot

    # ---------- helpers (not part of the interface but handy for viz) ----------

    @staticmethod
    def tile_value(log2_value: int) -> int:
        return 0 if log2_value == 0 else int(1 << log2_value)

    def max_tile(self, state: np.ndarray) -> int:
        return self.tile_value(int(state.max()))

    def score(self, state: np.ndarray) -> int:
        """Approximate cumulative score = sum of tile values present on the board.

        Not the true score of a playthrough (true score must be tracked by the
        caller via accumulated rewards); just a cheap viz aid.
        """
        return int(sum(self.tile_value(int(v)) for v in state.flatten()))
