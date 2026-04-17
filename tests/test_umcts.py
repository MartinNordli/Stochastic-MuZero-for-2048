"""Stochastic u-MCTS tests with a mocked AbstractStateManager.

The mock pretends action 2 ("left") leads to higher reward than the others
and the search should concentrate its visit count there after enough sims.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mcts.umcts import MCTSConfig, StochasticUMCTS


class MockASM:
    def __init__(self, num_actions=4, num_chance_codes=8, abstract_dim=4):
        self.num_actions = num_actions
        self.num_chance_codes = num_chance_codes
        self.abstract_dim = abstract_dim

    # --- pretend these are jit'd ----------------------------------------

    def phi(self, s, a):
        # A deterministic "afterstate" that tags which action produced it.
        sa = np.asarray(s, dtype=np.float32).copy()
        sa[0] = float(a)
        return sa

    def psi_logits(self, sa):
        # Uniform chance distribution, constant Q.
        q = 0.0
        sigma_logits = np.zeros(self.num_chance_codes, dtype=np.float32)
        return q, sigma_logits

    def dynamics(self, sa, c):
        # Reward depends only on the action used to produce sa (sa[0]); action
        # 2 (LEFT) yields a large reward, others yield small positive noise.
        a = int(sa[0])
        r = 10.0 if a == 2 else 0.1
        s_next = np.asarray(sa, dtype=np.float32) + 0.01 * c
        return s_next, r

    def predict_logits(self, s):
        # Uniform policy, zero value: PUCT must rely on Q from chance children.
        pi_logits = np.zeros(self.num_actions, dtype=np.float32)
        v = 0.0
        return pi_logits, v

    def predict(self, s):
        return np.ones(self.num_actions, dtype=np.float32) / self.num_actions, 0.0


def test_mcts_concentrates_visits_on_best_action():
    asm = MockASM()
    cfg = MCTSConfig(
        num_actions=4, num_chance_codes=8,
        M_s=200, d_max=5, discount=0.99,
        puct_c1=1.25, puct_c2=19652,
        root_dirichlet_alpha=0.3, root_exploration_frac=0.25,
    )
    rng = np.random.default_rng(0)
    search = StochasticUMCTS(asm, cfg, rng)

    root_s = np.zeros(asm.abstract_dim, dtype=np.float32)
    legal = np.ones(4, dtype=bool)
    pi, _v, root = search.run(root_s, legal)

    # Action 2 (LEFT) should receive the most visits since it alone yields
    # the large reward each step.
    assert int(np.argmax(pi)) == 2, f"expected action 2 to dominate, pi={pi}"
    assert pi[2] > 0.5, f"visits not concentrated: pi={pi}"


def test_mcts_respects_legal_mask():
    asm = MockASM()
    cfg = MCTSConfig(
        num_actions=4, num_chance_codes=8,
        M_s=50, d_max=3, discount=0.99,
        puct_c1=1.25, puct_c2=19652,
        root_dirichlet_alpha=0.3, root_exploration_frac=0.25,
    )
    rng = np.random.default_rng(0)
    search = StochasticUMCTS(asm, cfg, rng)

    root_s = np.zeros(asm.abstract_dim, dtype=np.float32)
    legal = np.array([True, True, False, True])  # disallow action 2
    pi, _v, _root = search.run(root_s, legal)
    assert pi[2] == 0.0, f"action 2 was masked out but got visits: pi={pi}"


def _run_all():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("\nmcts tests passed.")


if __name__ == "__main__":
    _run_all()
