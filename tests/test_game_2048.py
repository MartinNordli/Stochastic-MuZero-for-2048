"""Tests for the Game2048 SimWorld.

Runnable either with pytest or as `python tests/test_game_2048.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.games.game_2048 import DOWN, LEFT, RIGHT, UP, Game2048


def _board(rows):
    return np.array(rows, dtype=np.int8)


def test_slide_row_left_basic():
    g = Game2048()
    out, reward = g._slide_row_left(np.array([1, 1, 0, 0], dtype=np.int8))
    assert np.array_equal(out, [2, 0, 0, 0])
    assert reward == 4  # two '2's merged into a '4'


def test_slide_row_left_four_equal():
    g = Game2048()
    out, reward = g._slide_row_left(np.array([1, 1, 1, 1], dtype=np.int8))
    # 2048 merges pairwise greedily from the moving edge: [2,2,0,0].
    assert np.array_equal(out, [2, 2, 0, 0])
    assert reward == 8  # 4 + 4


def test_slide_row_left_three_equal():
    g = Game2048()
    out, reward = g._slide_row_left(np.array([1, 1, 1, 0], dtype=np.int8))
    # First two merge; third doesn't find a neighbour. [2,1,0,0].
    assert np.array_equal(out, [2, 1, 0, 0])
    assert reward == 4


def test_slide_row_left_chained_no_double_merge():
    g = Game2048()
    # [4,4,8,0] -> [8,8,0,0] after one merge at position 0; the produced '8'
    # must NOT then merge with the adjacent '8' on the same move.
    out, reward = g._slide_row_left(np.array([2, 2, 3, 0], dtype=np.int8))
    assert np.array_equal(out, [3, 3, 0, 0])
    assert reward == 8  # single merge of 4+4=8


def test_move_left_right_up_down_consistency():
    g = Game2048()
    s = _board([[0, 2, 0, 2], [1, 0, 1, 0], [0, 0, 3, 3], [4, 4, 0, 0]])

    left, lr, lc = g.move(s, LEFT)
    right, rr, rc = g.move(s, RIGHT)
    up, ur, uc = g.move(s, UP)
    down, dr, dc = g.move(s, DOWN)

    # Same total reward for left/right and for up/down (symmetry).
    assert lr == rr
    assert ur == dr
    # All four moves change the board here.
    assert lc and rc and uc and dc


def test_move_returns_changed_false_on_noop():
    g = Game2048()
    s = _board([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
    _, reward, changed = g.move(s, LEFT)
    assert reward == 0
    assert changed is False


def test_legal_actions_and_terminal():
    g = Game2048()
    dead = _board([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]])
    assert g.is_terminal(dead)
    assert not g.legal_actions(dead).any()

    alive = _board([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert not g.is_terminal(alive)
    assert g.legal_actions(alive)[LEFT]


def test_spawn_adds_one_tile_in_empty_cell():
    g = Game2048()
    rng = np.random.default_rng(0)
    s = _board([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    s2 = g.spawn(s, rng)
    assert (s2 != 0).sum() == 1
    placed_log2 = s2[s2 != 0][0]
    assert placed_log2 in (1, 2)


def test_step_illegal_action_is_noop():
    g = Game2048()
    rng = np.random.default_rng(0)
    s = _board([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
    # No empty cells and no merges possible on this full alternating board;
    # LEFT here is a no-op.
    s2, reward, done = g.step(s, LEFT, rng)
    assert np.array_equal(s, s2)
    assert reward == 0.0
    assert done is True


def test_initial_state_has_two_tiles():
    g = Game2048()
    rng = np.random.default_rng(0)
    s = g.initial_state(rng)
    assert (s != 0).sum() == 2


def test_encode_shape_and_onehot():
    g = Game2048(max_log2=11)
    s = _board([[0, 1, 2, 3], [11, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    enc = g.encode(s)
    assert enc.shape == (4, 4, 12)
    # Channel-sum is exactly 1 for every cell.
    assert np.allclose(enc.sum(axis=-1), 1.0)
    assert enc[0, 1, 1] == 1.0
    assert enc[1, 0, 11] == 1.0


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
