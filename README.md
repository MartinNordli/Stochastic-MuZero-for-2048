# Stochastic MuZero for 2048

A from-scratch implementation of **Stochastic MuZero** (Antonoglou et al., 2022) applied to **2048**. Built in **JAX + Flax linen + Optax**, with `jax.jit`-compiled network forwards, a `jax.vmap`-batched BPTT training step, and a matplotlib `FuncAnimation` of the trained agent.

NTNU IT-3105 *Kunstig Intelligens Programmering*, Spring 2025 — Main Project ("MuZero Knockoff"). The game 2048 is stochastic (random 2/4 tile spawns after every move), so plain MuZero isn't enough: we use the full Stochastic-MuZero architecture with afterstate and chance nodes.

---

## What's implemented

* **SimWorld for 2048** — separated deterministic `move` (slide + merge) from stochastic `spawn` (random 2/4 in an empty cell), so u-MCTS can traverse the split cleanly.
* **Five networks** (Flax linen):
    | Name  | Signature                            | Role                                   |
    |-------|--------------------------------------|----------------------------------------|
    | NN_r  | `board_stack → s`                    | observation → abstract state           |
    | NN_φ  | `(s, one_hot_action) → sᵃ`           | agent action in abstract space         |
    | NN_ψ  | `sᵃ → (Q, σ_logits)`                 | afterstate value + chance distribution |
    | NN_d  | `(sᵃ, one_hot_chance) → (s', r)`     | apply sampled chance outcome           |
    | NN_p  | `s → (π_logits, v)`                  | policy + value at decision node        |
* **Chance encoder** — VQ-VAE-style encoder with Gumbel-softmax + straight-through estimator; generates targets for NN_ψ's σ head from the real next observation at train time.
* **Stochastic u-MCTS** — alternating `DecisionNode` / `ChanceNode` traversal; PUCT at decision nodes, σ-sampling at chance nodes, reward-aware backup.
* **BPTT training step** — single `jax.value_and_grad` call through the composite graph, `jax.vmap` over the minibatch, jitted end-to-end. Loss = policy CE + value MSE + reward MSE + chance CE + afterstate MSE + L2.
* **Episode buffer** with windowed sampling for arbitrary lookback `q` and unroll horizon `w`.
* **Two inference modes**:
    * `play_actor` — NN_r + NN_p only (no MCTS), per the project spec's "deploy-time network".
    * `run_episode` — u-MCTS guided self-play.
* **Visualization** — canonical 2048 palette board renderer, per-step policy bar chart, `FuncAnimation` export to `.gif` / `.mp4`; plus a training-curves dashboard.

All pivotal parameters live in **one file**: `config/default.yaml`.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Smoke test — under a few minutes, proves the pipeline works
python scripts/train.py --config config/default.yaml \
    --N_e 15 --N_es 30 --M_s 8 --mbs 16 --I_t 3

# 2. Real training run (defaults from config/default.yaml — CPU-bound, hours)
python scripts/train.py --config config/default.yaml

# 3. Watch the trained agent
python scripts/play.py --checkpoint checkpoints/latest.pkl --viz
python scripts/play.py --checkpoint checkpoints/latest.pkl --save play.gif

# 4. Plot training curves
python scripts/visualize.py --log logs/train_log.jsonl
```

### CLI overrides

`scripts/train.py` accepts `--N_e --N_es --M_s --mbs --I_t --seed --train_steps_per_update` — any of these override the YAML. Useful for sweeping.

### Apple Silicon note

`scripts/train.py` forces `JAX_PLATFORMS=cpu`. The Metal backend is still missing ops for the network shapes used here; on CPU the nets are small enough to run comfortably.

---

## Project layout

```
Project 2 - Stochastic MuZero for 2048/
├── config/default.yaml                     # ALL pivotal parameters
├── requirements.txt
├── scripts/
│   ├── train.py                            # CLI: training loop
│   ├── play.py                             # CLI: watch/export a playthrough
│   └── visualize.py                        # CLI: plot training curves
├── src/
│   ├── games/
│   │   ├── game_base.py                    # abstract SimWorld interface
│   │   └── game_2048.py                    # 2048 SimWorld (move + spawn split)
│   ├── networks/
│   │   ├── networks.py                     # the five Flax modules
│   │   ├── chance_encoder.py               # Gumbel-softmax encoder + STE
│   │   └── losses.py                       # softmax CE, MSE, L2
│   ├── mcts/
│   │   ├── node.py                         # DecisionNode, ChanceNode
│   │   └── umcts.py                        # stochastic u-MCTS
│   ├── managers/
│   │   ├── game_state_manager.py           # GSM — wraps the SimWorld
│   │   ├── abstract_state_manager.py       # ASM — jit'd wrappers around the nets
│   │   ├── neural_network_manager.py       # NNM — BPTT + checkpoints
│   │   └── rl_manager.py                   # RLM — self-play + deploy actor
│   ├── training/
│   │   ├── episode_buffer.py               # EB — windowed minibatch sampling
│   │   └── train.py                        # EPISODE_LOOP
│   ├── viz/
│   │   ├── board_renderer.py               # canonical 2048 palette
│   │   ├── play_animation.py               # FuncAnimation(board + π bar chart)
│   │   └── training_plots.py               # 2×3 diagnostics dashboard
│   └── util/config.py                      # YAML → dataclass loader
├── tests/
│   ├── test_game_2048.py                   # slide/merge, legal actions, terminal
│   └── test_umcts.py                       # visit counts + legal-mask respect
├── checkpoints/                            # .pkl snapshots (saved during training)
└── logs/                                   # JSONL training log + exported plots
```

---

## Configuration

Everything that matters is in `config/default.yaml`:

```yaml
seed: 42
game:     {grid_size: 4, spawn_two_prob: 0.9, max_log2: 15}
search:   {M_s: 50, d_max: 5, discount: 0.997,
           puct_c1: 1.25, puct_c2: 19652,
           root_dirichlet_alpha: 0.3, root_exploration_frac: 0.25,
           num_chance_codes: 32}
network:  {abstract_dim: 128, repr_channels: 64,
           dynamics_hidden: 128, afterstate_hidden: 128,
           prediction_hidden: 128, l2: 1.0e-4,
           gumbel_temperature: 1.0}
training: {N_e: 2000, N_es: 500, I_t: 10, mbs: 64, q: 0, w: 5,
           lr: 2.0e-3, lr_decay: 0.9,
           temperature_schedule: [[0, 1.0], [500, 0.5], [1500, 0.1]]}
logging:  {log_dir: "logs/", checkpoint_dir: "checkpoints/",
           save_interval: 50, viz_flag: false}
```

Knob cheatsheet:

| Knob                        | Effect                                          |
|-----------------------------|-------------------------------------------------|
| `search.M_s`                | MCTS simulations per move (speed ↔ quality)     |
| `search.d_max`              | rollout depth after expansion                   |
| `search.num_chance_codes`   | size of abstract chance space                   |
| `training.w`                | BPTT unroll horizon                             |
| `training.q`                | observation lookback (0 = only current board)   |
| `training.I_t`              | episodes between training passes                |
| `training.mbs`              | minibatch size                                  |
| `training.temperature_schedule` | piecewise-constant exploration temperature   |

---

## Tests

Pure-Python tests (no pytest required):

```bash
python tests/test_game_2048.py
python tests/test_umcts.py
```

---

## Outputs

* `logs/train_log.jsonl` — one JSON record per episode: `episode, length, reward, max_tile, temperature, buffer_size, elapsed, loss{total, policy, value, reward, chance, afterstate, l2}`.
* `checkpoints/ckpt_ep{N}.pkl` — snapshots every `save_interval` episodes (plus `latest.pkl`).
* `logs/curves.png`, `logs/playthrough.gif` — written by `scripts/visualize.py` and `scripts/play.py --save …`.

---

## Design notes

* **Why Stochastic MuZero, not plain MuZero?** 2048's tile spawn is a genuine stochastic branch. Plain MuZero collapses it into a deterministic latent transition and pays for it in learned dynamics noise. Stochastic MuZero makes the branch explicit: NN_ψ predicts a categorical over chance codes, NN_d consumes the sampled code — chance space is abstract and fixed-size (`num_chance_codes`), so we never enumerate 2048's real spawn possibilities.
* **Chance targets.** At train time we encode `(sᵃ, NN_r(real_next_board))` into a soft chance code via Gumbel-softmax; NN_ψ's σ head is trained to match that code via CE with `stop_gradient` on the target. The straight-through step ensures NN_d sees a hard one-hot on the forward pass while gradients flow through the soft distribution.
* **Deterministic / stochastic split.** `Game2048.move` returns `(afterstate, reward, changed)`, and `Game2048.spawn` samples the tile. u-MCTS walks the split directly: decision → chance (via NN_φ) → decision (via NN_d). At self-play time we just call `step = move → spawn`.
* **Deploy-time actor.** Per the project spec, the trained policy should be usable *without* MCTS. `RLManager.play_actor` does exactly that — it threads NN_r → NN_p only, returning a policy-gradient-style actor.

### Departures from the given pseudocode

The course pseudocode describes classic MuZero (`NN_d: (σ, a) → (σ', r)`). For Stochastic MuZero this splits into:

* `NN_φ: (s, a) → sᵃ`
* `NN_ψ: sᵃ → (Q, σ)`
* `NN_d: (sᵃ, c) → (s', r)`

plus the chance encoder (used only at training time). `DO_ROLLOUT` and `DO_BACKPROPAGATION` generalize to alternating decision/chance node types; the BPTT unroll alternates `NN_φ` and `NN_d`. The episode loop, buffer, and training cadence otherwise match the spec.

---

## References

* Antonoglou et al., *Planning in Stochastic Environments with a Learned Model*, ICLR 2022.
* Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero), Nature 2020.
