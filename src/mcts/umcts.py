"""Stochastic u-MCTS (MuZero's MCTS extended with afterstate + chance nodes).

Follows the project pseudocode's shape:
    For m in range(M_s):
        * Tree-policy walk from root to a leaf decision node L.
        * Expand L: create all A chance children (via NN_phi + NN_psi).
        * Pick a random chance child c*.
        * DO_ROLLOUT from c* for (d_max - depth) steps, accumulating a
          discounted return plus a bootstrap value at the horizon.
        * DO_BACKPROPAGATION: update visit counts and value sums back to the
          root, respecting the per-edge rewards emitted on NN_d transitions.

Two node types (src/mcts/node.py):
    DecisionNode  — holds abstract state s and action-indexed chance children.
    ChanceNode    — holds abstract afterstate sa and code-indexed decision
                    children; sigma is its chance distribution from NN_psi.

The tree is plain Python data (not a JAX pytree). Only the network forwards
called through the ASM are jit-compiled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .node import ChanceNode, DecisionNode


# ---------------------------------------------------------------- helpers


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    e = np.exp(x - m)
    return (e / e.sum()).astype(np.float32)


def _puct_select(node: DecisionNode, c1: float, c2: float) -> int:
    total = max(sum(ch.visits for ch in node.children.values()), 1)
    log_term = np.log((total + c2 + 1) / c2)
    scale = c1 + log_term
    best_a, best_score = -1, -np.inf
    for a, chance in node.children.items():
        q = chance.value
        u = scale * float(node.prior[a]) * np.sqrt(total) / (1 + chance.visits)
        score = q + u
        if score > best_score:
            best_score, best_a = score, a
    return best_a


def _sample(rng: np.random.Generator, p: np.ndarray) -> int:
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def _add_exploration_noise(
    rng: np.random.Generator, prior: np.ndarray, alpha: float, frac: float
) -> np.ndarray:
    noise = rng.dirichlet([alpha] * len(prior)).astype(np.float32)
    return ((1 - frac) * prior + frac * noise).astype(np.float32)


# ----------------------------------------------------------------- config


@dataclass
class MCTSConfig:
    num_actions: int
    num_chance_codes: int
    M_s: int
    d_max: int
    discount: float
    puct_c1: float
    puct_c2: float
    root_dirichlet_alpha: float
    root_exploration_frac: float


# ---------------------------------------------------------------- searcher


class StochasticUMCTS:
    """Run M_s simulations and return a root action distribution + value.

    `asm` is an AbstractStateManager instance that supplies (jit'd) forwards:
        phi(s, a) -> sa
        psi_logits(sa) -> (q, sigma_logits)
        dynamics(sa, c) -> (s', r)
        predict_logits(s) -> (pi_logits, v)
        predict(s) -> (pi_probs, v)
    """

    def __init__(self, asm, cfg: MCTSConfig, rng: np.random.Generator):
        self.asm = asm
        self.cfg = cfg
        self.rng = rng

    # -------- public entry ------------------------------------------------

    def run(self, root_s: np.ndarray, legal_mask: np.ndarray):
        root = DecisionNode(s=root_s)
        self._expand_decision(root, legal_mask=legal_mask, add_noise=True)

        for _ in range(self.cfg.M_s):
            self._simulate(root)

        visits = np.array(
            [root.children[a].visits if a in root.children else 0
             for a in range(self.cfg.num_actions)],
            dtype=np.float32,
        )
        total = visits.sum()
        pi = visits / total if total > 0 else np.ones_like(visits) / len(visits)
        return pi, float(root.value), root

    # -------- single simulation ------------------------------------------

    def _simulate(self, root: DecisionNode):
        # path: list of (node, r_on_incoming_edge). Reward is the reward
        # emitted by NN_d when this node was produced from its chance parent
        # (0 for chance children of decision nodes — dec->chance edges are
        # "agent action" edges and carry no reward).
        path: List[Tuple[object, float]] = [(root, 0.0)]
        node = root
        depth = 0

        # 1. Selection: walk down, lazily creating decision grandchildren at
        # chance nodes, until we hit an unexpanded decision leaf.
        while node.is_expanded and depth < self.cfg.d_max:
            a = _puct_select(node, self.cfg.puct_c1, self.cfg.puct_c2)
            chance = node.children[a]
            path.append((chance, 0.0))

            c = _sample(self.rng, chance.sigma)
            if c not in chance.children:
                s_next, r = self.asm.dynamics(chance.sa, c)
                chance.children[c] = DecisionNode(
                    s=np.asarray(s_next), reward_from_parent=float(r)
                )
            dec_child = chance.children[c]
            path.append((dec_child, dec_child.reward_from_parent))
            node = dec_child
            depth += 1

        # 2. Expand the leaf decision node (if we haven't run out of depth).
        if not node.is_expanded and depth < self.cfg.d_max:
            self._expand_decision(node, legal_mask=None, add_noise=False)

        # 3. Pick a random chance child of the leaf (if any exist) and start
        # the rollout from it; otherwise bootstrap with NN_p at the leaf.
        if node.children:
            a_star = int(self.rng.integers(0, self.cfg.num_actions))
            # If a_star got masked out at the root (only applies when legal_mask
            # was supplied); for deeper leaves legal_mask is None and all A are
            # present. Fall back to any available child.
            if a_star not in node.children:
                a_star = next(iter(node.children))
            c_star = node.children[a_star]
            path.append((c_star, 0.0))
            leaf_value = self._rollout(c_star, remaining_depth=self.cfg.d_max - depth)
        else:
            _, v = self.asm.predict(node.s)
            leaf_value = float(v)

        # 4. Backprop. g starts as the estimated return from the last node on
        # the path. Going up one level: G(parent) = r_incoming_to_child + γ · G(child).
        g = leaf_value
        for i in range(len(path) - 1, -1, -1):
            node_i, r_i = path[i]
            node_i.visits += 1
            node_i.value_sum += g
            # Move g "up" one level for the next iteration.
            g = r_i + self.cfg.discount * g

    # -------- rollout -----------------------------------------------------

    def _rollout(self, c_star: ChanceNode, remaining_depth: int) -> float:
        """Simulate forward using the learned model (NN_d + NN_p + NN_phi +
        NN_psi), return discounted sum of rewards + bootstrap value."""
        discount = self.cfg.discount
        acc = 0.0
        disc = 1.0

        # First step out of c_star: sample a code + apply NN_d.
        c = _sample(self.rng, c_star.sigma)
        s, r = self.asm.dynamics(c_star.sa, c)
        acc += disc * float(r)
        disc *= discount
        remaining = max(remaining_depth - 1, 0)

        for _ in range(remaining):
            pi_logits, _ = self.asm.predict_logits(s)
            pi = _softmax(np.asarray(pi_logits))
            a = int(self.rng.choice(self.cfg.num_actions, p=pi))
            sa = self.asm.phi(s, a)
            _, sigma_logits = self.asm.psi_logits(sa)
            sigma = _softmax(np.asarray(sigma_logits))
            c = int(self.rng.choice(self.cfg.num_chance_codes, p=sigma))
            s, r = self.asm.dynamics(sa, c)
            acc += disc * float(r)
            disc *= discount

        _, v = self.asm.predict(s)
        acc += disc * float(v)
        return acc

    # -------- expansion ---------------------------------------------------

    def _expand_decision(
        self,
        node: DecisionNode,
        legal_mask: np.ndarray | None,
        add_noise: bool,
    ):
        pi_logits, _ = self.asm.predict_logits(node.s)
        prior = _softmax(np.asarray(pi_logits))
        if legal_mask is not None:
            prior = prior * legal_mask.astype(prior.dtype)
            s = prior.sum()
            prior = prior / s if s > 0 else np.ones_like(prior) / len(prior)
        if add_noise:
            prior = _add_exploration_noise(
                self.rng, prior,
                self.cfg.root_dirichlet_alpha,
                self.cfg.root_exploration_frac,
            )
        node.prior = prior
        for a in range(self.cfg.num_actions):
            if legal_mask is not None and not legal_mask[a]:
                continue
            sa = self.asm.phi(node.s, a)
            _, sigma_logits = self.asm.psi_logits(sa)
            sigma = _softmax(np.asarray(sigma_logits))
            node.children[a] = ChanceNode(sa=np.asarray(sa), sigma=sigma)
        node.is_expanded = True
