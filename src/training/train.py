"""EPISODE_LOOP orchestrator.

Wires together the GSM, ASM, RLM, NNM, MCTS config, and EpisodeBuffer, then
drives the outer training loop as described in the project pseudocode:

    EH = []
    init Psi
    for ep in range(N_e):
        epidata = run_episode(...)
        EH.append(epidata)
        if ep % I_t == 0:
            DO_BPTT_TRAINING(Psi, EH, mbs)
    return Psi
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import numpy as np

from src.games.game_2048 import Game2048
from src.managers.abstract_state_manager import AbstractStateManager
from src.managers.game_state_manager import GameStateManager
from src.managers.neural_network_manager import NeuralNetworkManager
from src.managers.rl_manager import RLManager
from src.mcts.umcts import MCTSConfig
from src.networks.networks import MuZeroNets
from src.training.episode_buffer import EpisodeBuffer
from src.util.config import Config


def build_system(cfg: Config):
    """Factory that assembles every component from the config."""
    np_rng = np.random.default_rng(cfg.seed)
    jax_rng = jax.random.PRNGKey(cfg.seed)

    world = Game2048(
        grid_size=cfg.game.grid_size,
        spawn_two_prob=cfg.game.spawn_two_prob,
        max_log2=cfg.game.max_log2,
    )
    gsm = GameStateManager(world, np_rng)

    nets = MuZeroNets(
        num_actions=world.num_actions,
        num_chance_codes=cfg.search.num_chance_codes,
        abstract_dim=cfg.network.abstract_dim,
        repr_channels=cfg.network.repr_channels,
        afterstate_hidden=cfg.network.afterstate_hidden,
        dynamics_hidden=cfg.network.dynamics_hidden,
        prediction_hidden=cfg.network.prediction_hidden,
    )
    nnm = NeuralNetworkManager(
        nets=nets,
        grid_size=cfg.game.grid_size,
        max_log2=cfg.game.max_log2,
        w=cfg.training.w,
        q=cfg.training.q,
        lr=cfg.training.lr,
        l2=cfg.network.l2,
        gumbel_temperature=cfg.network.gumbel_temperature,
        num_actions=world.num_actions,
        num_chance_codes=cfg.search.num_chance_codes,
    )
    params, enc_params, opt_state = nnm.init_params(jax_rng)

    asm = AbstractStateManager(
        nets=nets,
        num_chance_codes=cfg.search.num_chance_codes,
        num_actions=world.num_actions,
    )
    asm.set_params(params)

    mcts_cfg = MCTSConfig(
        num_actions=world.num_actions,
        num_chance_codes=cfg.search.num_chance_codes,
        M_s=cfg.search.M_s,
        d_max=cfg.search.d_max,
        discount=cfg.search.discount,
        puct_c1=cfg.search.puct_c1,
        puct_c2=cfg.search.puct_c2,
        root_dirichlet_alpha=cfg.search.root_dirichlet_alpha,
        root_exploration_frac=cfg.search.root_exploration_frac,
    )

    rlm = RLManager(
        gsm=gsm,
        asm=asm,
        mcts_cfg=mcts_cfg,
        max_steps=cfg.training.N_es,
        q=cfg.training.q,
        rng=np_rng,
    )

    eb = EpisodeBuffer(capacity=max(cfg.training.I_t * 20, 200))

    return {
        "cfg": cfg,
        "world": world,
        "gsm": gsm,
        "nets": nets,
        "nnm": nnm,
        "asm": asm,
        "rlm": rlm,
        "eb": eb,
        "params": params,
        "enc_params": enc_params,
        "opt_state": opt_state,
        "np_rng": np_rng,
        "jax_rng": jax_rng,
    }


def episode_loop(cfg: Config, train_steps_per_update: int = 10):
    """Main training loop."""
    sys_ = build_system(cfg)
    cfg = sys_["cfg"]
    rlm = sys_["rlm"]
    nnm = sys_["nnm"]
    asm = sys_["asm"]
    eb = sys_["eb"]
    np_rng = sys_["np_rng"]

    params = sys_["params"]
    enc_params = sys_["enc_params"]
    opt_state = sys_["opt_state"]

    log_dir = Path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_log.jsonl"
    ckpt_dir = Path(cfg.logging.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rng_key = jax.random.PRNGKey(cfg.seed + 1)
    t0 = time.time()

    for ep in range(cfg.training.N_e):
        asm.set_params(params)
        temperature = cfg.temperature_at(ep)

        episode = rlm.run_episode(temperature)
        eb.add(episode)
        ep_reward = float(episode.rewards.sum())
        ep_len = int(len(episode))
        max_tile = sys_["world"].max_tile(episode.states[-1])

        losses = None
        if ep > 0 and ep % cfg.training.I_t == 0 and eb.total_steps() >= cfg.training.mbs:
            for _ in range(train_steps_per_update):
                batch = eb.sample_batch(
                    np_rng, cfg.training.mbs, cfg.training.q, cfg.training.w,
                    num_actions=sys_["world"].num_actions,
                )
                rng_key, sub = jax.random.split(rng_key)
                params, enc_params, opt_state, losses = nnm.train_step(
                    params, enc_params, opt_state, batch, sub,
                )

        record = {
            "episode": ep,
            "length": ep_len,
            "reward": ep_reward,
            "max_tile": max_tile,
            "temperature": temperature,
            "buffer_size": len(eb),
            "elapsed": time.time() - t0,
            "loss": {k: float(v) for k, v in losses.items()} if losses is not None else None,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        line = (f"ep {ep:4d}  len={ep_len:4d}  R={ep_reward:7.1f}  "
                f"maxtile={max_tile:5d}  T={temperature:.2f}")
        if losses is not None:
            line += (f"  L={float(losses['total']):.3f}"
                     f"  pol={float(losses['policy']):.2f}"
                     f"  val={float(losses['value']):.2f}"
                     f"  rew={float(losses['reward']):.2f}"
                     f"  ch={float(losses['chance']):.2f}")
        print(line, flush=True)

        if ep > 0 and ep % cfg.logging.save_interval == 0:
            nnm.save(
                ckpt_dir / f"ckpt_ep{ep:06d}.pkl",
                params, enc_params, opt_state,
                meta={"episode": ep, "cfg": cfg.raw},
            )
            nnm.save(
                ckpt_dir / "latest.pkl",
                params, enc_params, opt_state,
                meta={"episode": ep, "cfg": cfg.raw},
            )

    # Final checkpoint.
    nnm.save(
        ckpt_dir / "latest.pkl",
        params, enc_params, opt_state,
        meta={"episode": cfg.training.N_e - 1, "cfg": cfg.raw},
    )
    return params, enc_params, opt_state
