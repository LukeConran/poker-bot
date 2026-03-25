"""
Curriculum DQN training loop using RLCard's No-Limit Hold'em environment.

Phase 1: DQN trains against a RandomAgent — easy opponent, fast initial learning
Phase 2: DQN fine-tunes against our PokerBotAgent — stronger baseline

Checkpoints are saved to results/checkpoints/ after each phase.
"""

import json
import logging
import os
import time
import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import set_seed, reorganize

logging.getLogger("rlcard").setLevel(logging.WARNING)

from rlcard_agent import PokerBotAgent


RESULTS_DIR    = "results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
SEED           = 42

# Phase 1: beat random first
PHASE1_HANDS   = 50000

# Phase 2: fine-tune against probability bot
PHASE2_HANDS   = 20000

EVAL_EVERY     = 2000  # evaluate every N hands within each phase
EVAL_HANDS     = 500   # hands per evaluation
SAVE_EVERY     = 5000  # checkpoint every N hands


def make_dqn(env, total_hands: int) -> DQNAgent:
    return DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        learning_rate=5e-4,
        batch_size=64,
        replay_memory_size=20000,
        replay_memory_init_size=500,
        train_every=1,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=total_hands * 3,
    )


def make_envs(seed: int):
    cfg = {"num_players": 2, "allow_raw_data": True}
    train_env = rlcard.make("no-limit-holdem", config={**cfg, "seed": seed})
    eval_env  = rlcard.make("no-limit-holdem", config={**cfg, "seed": seed + 1})
    return train_env, eval_env


def evaluate(eval_env, dqn, opponent) -> float:
    """Run EVAL_HANDS hands without training, return DQN avg payoff."""
    eval_env.set_agents([dqn, opponent])
    payoffs = []
    for _ in range(EVAL_HANDS):
        _, p = eval_env.run(is_training=False)
        payoffs.append(p[0])
    return sum(payoffs) / len(payoffs)


def train_phase(phase: int, dqn, train_env, eval_env, opponent, num_hands: int, hand_log: list, start: float):
    label = "RandomAgent" if phase == 1 else "PokerBotAgent"
    print(f"\n=== Phase {phase}: {num_hands} hands vs {label} ===")

    train_env.set_agents([dqn, opponent])

    for hand_num in range(1, num_hands + 1):
        trajectories, payoffs = train_env.run(is_training=True)
        transitions = reorganize(trajectories, payoffs)
        for ts in transitions[0]:
            dqn.feed(ts)

        hand_log.append({
            "phase": phase,
            "hand": hand_num,
            "payoffs": [float(p) for p in payoffs],
        })

        if hand_num % EVAL_EVERY == 0:
            eval_opponent = RandomAgent(num_actions=eval_env.num_actions) if phase == 1 else PokerBotAgent()
            avg = evaluate(eval_env, dqn, eval_opponent)
            elapsed = time.time() - start
            print(
                f"  Hand {hand_num:>{len(str(num_hands))}} | "
                f"avg payoff vs {label} ({EVAL_HANDS} hands): {avg:+.2f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        if hand_num % SAVE_EVERY == 0:
            fname = f"dqn_phase{phase}_{hand_num}.pt"
            dqn.save_checkpoint(CHECKPOINT_DIR, filename=fname)
            print(f"  Checkpoint saved: {CHECKPOINT_DIR}/{fname}")


def run():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(SEED)

    train_env, eval_env = make_envs(SEED)
    dqn = make_dqn(train_env, PHASE1_HANDS + PHASE2_HANDS)

    hand_log = []
    start = time.time()

    # Phase 1 — train against random
    train_phase(1, dqn, train_env, eval_env, RandomAgent(num_actions=train_env.num_actions), PHASE1_HANDS, hand_log, start)
    dqn.save_checkpoint(CHECKPOINT_DIR, filename="dqn_phase1_complete.pt")
    print(f"\nPhase 1 complete. Checkpoint saved.")

    # Phase 2 — fine-tune against probability bot
    train_phase(2, dqn, train_env, eval_env, PokerBotAgent(), PHASE2_HANDS, hand_log, start)
    dqn.save_checkpoint(CHECKPOINT_DIR, filename="dqn_final.pt")
    print(f"\nPhase 2 complete. Final checkpoint saved.")

    # Save hand log
    log_path = os.path.join(RESULTS_DIR, "hand_log.json")
    with open(log_path, "w") as f:
        json.dump(hand_log, f, indent=2)

    # Summary
    total = len(hand_log)
    wins  = sum(1 for h in hand_log if h["payoffs"][0] > 0)
    print(f"\n--- Summary ({total} total hands) ---")
    print(f"Overall DQN win rate : {wins / total:.1%}")
    print(f"Final checkpoint     : {CHECKPOINT_DIR}/dqn_final.pt")
    print(f"Hand log             : {log_path}")


if __name__ == "__main__":
    run()
