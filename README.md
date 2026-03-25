# Poker Bot

A No-Limit Texas Hold'em bot built in two layers: a probability-based decision engine for structured reasoning, and a Neural Fictitious Self-Play (NFSP) agent trained via self-play to approach Nash equilibrium.

## How It Works

### Layer 1 — Probability Bot (`bot.py`)

The foundation is a rule-based bot that makes every decision using two primitives:

- **Hand equity** — Monte Carlo simulation over random runouts to estimate win probability
- **Pot odds** — expected value calculation to determine if calling or raising is profitable

It serves as both a standalone agent and a training benchmark for the learned bot.

### Layer 2 — NFSP Agent (`train_nfsp.py`)

The trained agent uses **Neural Fictitious Self-Play**, an algorithm that maintains two neural networks simultaneously:

- A **best-response network** (RL, like DQN) that learns to exploit the current opponent
- An **average strategy network** (supervised learning) that tracks the historical average policy

The average strategy converges toward Nash equilibrium in two-player zero-sum games, making it theoretically unexploitable. Training uses self-play: two NFSP agents play against each other, both updating every hand.

## Project Structure

```
├── bot.py              # Probability-based decision logic
├── game_state.py       # GameState dataclass (hole cards, board, pot, stack, etc.)
├── equity.py           # Monte Carlo equity estimator
├── hand_evaluator.py   # Wrapper around treys for hand evaluation
├── pot_odds.py         # Pot odds, EV, and raise sizing calculations
├── rlcard_agent.py     # Wraps Bot in RLCard's agent interface
├── train.py            # DQN curriculum training (deprecated)
├── train_nfsp.py       # NFSP self-play training (current)
├── play.py             # Interactive play against the trained bot
├── STORY.md            # Narrative development log
└── results/
    └── nfsp_checkpoints/   # Saved model checkpoints (.pt files)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install rlcard torch treys
```

## Training

```bash
python train_nfsp.py
```

Training runs in two phases:

| Phase | Hands | Opponent | Purpose |
|-------|-------|----------|---------|
| 1 (warm-up) | 10k × 2 | RandomAgent | Bootstrap basic hand strength |
| 2 (self-play) | 1,000,000 | Other NFSP agent | Converge toward Nash equilibrium |

Checkpoints are saved to `results/nfsp_checkpoints/` every 50,000 hands. Evaluation (avg payoff vs the probability bot) is printed every 25,000 hands.

On a laptop CPU, expect roughly 100–200 hands/second — a 1M-hand run takes 2–3 hours.

## Playing Against the Bot

```bash
python play.py
python play.py --checkpoint results/nfsp_checkpoints/nfsp_phase2_500000.pt
python play.py --chips 200
```

Each session starts with equal stacks. The env is recreated each hand with the correct effective stack so neither player can bet more than they have. Play continues until someone busts or you quit.

**Action shortcuts:**

| Key | Action |
|-----|--------|
| F | Fold |
| C | Check / Call |
| H | Raise half pot |
| P | Raise pot |
| A | All-in |

## Algorithm Comparison

| Algorithm | Nash convergence | Compute | Notes |
|-----------|-----------------|---------|-------|
| DQN | No | Low | Struggles with hidden information |
| NFSP | Yes (2-player) | Low–Medium | Current approach |
| DMC | No | Medium | Better for multi-player pools |
| CFR | Yes | Very high | Requires game abstraction at scale |

## References

- Zha et al. (2019). *RLCard: A Toolkit for Reinforcement Learning in Card Games.* IJCAI-20. <https://github.com/datamllab/rlcard>
- Heinrich & Silver (2016). *Deep Reinforcement Learning from Self-Play in Imperfect-Information Games.* arXiv:1603.01121
