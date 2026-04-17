"""Load the YAML config into a typed, dot-accessible object.

Keeps the "pivotal parameters live in ONE place" rule: the rest of the code
takes a single `Config` instance and reads from it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class GameCfg:
    name: str
    grid_size: int
    spawn_two_prob: float
    max_log2: int


@dataclass
class SearchCfg:
    M_s: int
    d_max: int
    discount: float
    puct_c1: float
    puct_c2: float
    root_dirichlet_alpha: float
    root_exploration_frac: float
    num_chance_codes: int


@dataclass
class NetworkCfg:
    abstract_dim: int
    repr_channels: int
    dynamics_hidden: int
    afterstate_hidden: int
    prediction_hidden: int
    l2: float
    gumbel_temperature: float


@dataclass
class TrainingCfg:
    N_e: int
    N_es: int
    I_t: int
    mbs: int
    q: int
    w: int
    lr: float
    lr_decay: float
    temperature_schedule: List[Tuple[int, float]]


@dataclass
class LoggingCfg:
    log_dir: str
    checkpoint_dir: str
    save_interval: int
    eval_interval: int
    viz_flag: bool


@dataclass
class Config:
    seed: int
    game: GameCfg
    search: SearchCfg
    network: NetworkCfg
    training: TrainingCfg
    logging: LoggingCfg
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            seed=int(raw["seed"]),
            game=GameCfg(**raw["game"]),
            search=SearchCfg(**raw["search"]),
            network=NetworkCfg(**raw["network"]),
            training=TrainingCfg(
                **{**raw["training"],
                   "temperature_schedule": [tuple(p) for p in raw["training"]["temperature_schedule"]]},
            ),
            logging=LoggingCfg(**raw["logging"]),
            raw=raw,
        )

    def temperature_at(self, episode: int) -> float:
        """Piecewise-constant temperature schedule keyed on the episode index."""
        schedule = self.training.temperature_schedule
        current = schedule[0][1]
        for ep, temp in schedule:
            if episode >= ep:
                current = temp
            else:
                break
        return current
