"""Tree nodes for stochastic u-MCTS.

DecisionNode: agent picks an action (PUCT selection over priors).
ChanceNode:   environment samples a chance code ~ sigma (stochastic branch).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class ChanceNode:
    """Stores abstract afterstate (post-action, pre-chance)."""
    sa: np.ndarray
    sigma: np.ndarray                          # (K,) probability vector from NN_psi
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[int, "DecisionNode"] = field(default_factory=dict)  # code -> decision grandchild

    @property
    def value(self) -> float:
        return self.value_sum / max(self.visits, 1)


@dataclass
class DecisionNode:
    """Stores abstract state."""
    s: np.ndarray
    prior: Optional[np.ndarray] = None         # (num_actions,) softmaxed from NN_p
    reward_from_parent: float = 0.0            # reward emitted on the NN_d edge that led here
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[int, ChanceNode] = field(default_factory=dict)   # action -> chance child
    is_expanded: bool = False

    @property
    def value(self) -> float:
        return self.value_sum / max(self.visits, 1)
