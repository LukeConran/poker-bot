"""
NFSP self-play training loop using RLCard's No-Limit Hold'em environment.

Two NFSPAgent instances (nfsp0, nfsp1) train together via self-play, converging
toward Nash equilibrium. nfsp0 is the agent we care about; nfsp1 is its sparring
partner.

Phase 1: nfsp0 warms up against a RandomAgent — bootstraps basic poker quickly.
Phase 2: Both agents play each other (self-play) — the core NFSP training regime.

Evaluation measures nfsp0's avg payoff vs PokerBotAgent as a meaningful benchmark.
Checkpoints saved to results/nfsp_checkpoints/ after each phase.
"""

import json
import logging
import os
import time
import rlcard
from rlcard.agents import NFSPAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import set_seed, reorganize

logging.basicConfig(level=logging.WARNING)
logging.getLogger("rlcard").setLevel(logging.WARNING)

from rlcard_agent import PokerBotAgent


RESULTS_DIR    = "results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "nfsp_checkpoints")
SEED           = 42

# Phase 1: brief warm-up vs random — just enough to learn basic hand strength
PHASE1_HANDS   = 10000

# Phase 2: self-play — the main training regime
PHASE2_HANDS   = 1_000_000

EVAL_EVERY     = 25000  # evaluate every N hands within each phase
EVAL_HANDS     = 1000   # hands per evaluation episode
SAVE_EVERY     = 50000  # checkpoint every N hands


def make_nfsp(env, total_hands: int) -> NFSPAgent:
    return NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[64, 64],
        reservoir_buffer_capacity=20000,
        anticipatory_param=0.1,          # 10% best-response, 90% avg policy
        batch_size=256,
        train_every=1,
        sl_learning_rate=5e-3,
        min_buffer_size_to_learn=100,
        q_replay_memory_size=20000,
        q_replay_memory_init_size=100,
        q_epsilon_start=0.06,
        q_epsilon_end=0.0,
        q_epsilon_decay_steps=total_hands * 3,
        q_batch_size=32,
        q_train_every=1,
        q_mlp_layers=[64, 64],
        rl_learning_rate=5e-4,
        evaluate_with="average_policy",  # use avg policy at eval time (unexploitable)
    )


def make_envs(seed: int):
    cfg = {"num_players": 2, "allow_raw_data": True}
    train_env = rlcard.make("no-limit-holdem", config={**cfg, "seed": seed})
    eval_env  = rlcard.make("no-limit-holdem", config={**cfg, "seed": seed + 1})
    return train_env, eval_env


def evaluate(eval_env, nfsp, opponent) -> float:
    """Run EVAL_HANDS hands without training, return nfsp avg payoff."""
    eval_env.set_agents([nfsp, opponent])
    payoffs = []
    for _ in range(EVAL_HANDS):
        _, p = eval_env.run(is_training=False)
        payoffs.append(p[0])
    return sum(payoffs) / len(payoffs)


def train_phase(
    phase: int,
    nfsp0: NFSPAgent,
    nfsp1: NFSPAgent,
    train_env,
    eval_env,
    num_hands: int,
    hand_log: list,
    start: float,
):
    """
    Phase 1: nfsp0 trains against a fixed RandomAgent. Only nfsp0 is updated.
    Phase 2: nfsp0 and nfsp1 play each other. Both agents are updated each hand.
    Evaluation is always nfsp0 vs PokerBotAgent (fixed benchmark).
    """
    if phase == 1:
        label    = "RandomAgent (warm-up)"
        opponent = RandomAgent(num_actions=train_env.num_actions)
    else:
        label    = "Self-play"
        opponent = nfsp1

    print(f"\n=== Phase {phase}: {num_hands} hands — {label} ===")
    train_env.set_agents([nfsp0, opponent])

    for hand_num in range(1, num_hands + 1):
        trajectories, payoffs = train_env.run(is_training=True)
        transitions = reorganize(trajectories, payoffs)

        # Always update nfsp0 (seat 0)
        for ts in transitions[0]:
            nfsp0.feed(ts)

        # In self-play, also update nfsp1 (seat 1)
        if phase == 2:
            for ts in transitions[1]:
                nfsp1.feed(ts)

        hand_log.append({
            "phase": phase,
            "hand": hand_num,
            "payoffs": [float(p) for p in payoffs],
        })

        if hand_num % EVAL_EVERY == 0:
            avg = evaluate(eval_env, nfsp0, PokerBotAgent())
            elapsed = time.time() - start
            print(
                f"  Hand {hand_num:>{len(str(num_hands))}} | "
                f"avg payoff vs PokerBot ({EVAL_HANDS} hands): {avg:+.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        if hand_num % SAVE_EVERY == 0:
            fname = f"nfsp_phase{phase}_{hand_num}.pt"
            nfsp0.save_checkpoint(CHECKPOINT_DIR, filename=fname)
            print(f"  Checkpoint saved: {CHECKPOINT_DIR}/{fname}")


def run():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(SEED)

    train_env, eval_env = make_envs(SEED)
    total_hands = PHASE1_HANDS + PHASE2_HANDS
    nfsp0 = make_nfsp(train_env, total_hands)
    nfsp1 = make_nfsp(train_env, total_hands)

    hand_log = []
    start = time.time()

    # Phase 1 — warm up both agents vs random, alternating seats
    train_phase(1, nfsp0, nfsp1, train_env, eval_env, PHASE1_HANDS, hand_log, start)
    train_phase(1, nfsp1, nfsp0, train_env, eval_env, PHASE1_HANDS, hand_log, start)
    nfsp0.save_checkpoint(CHECKPOINT_DIR, filename="nfsp_phase1_complete.pt")
    print("\nPhase 1 complete. Checkpoint saved.")

    # Phase 2 — self-play
    train_phase(2, nfsp0, nfsp1, train_env, eval_env, PHASE2_HANDS, hand_log, start)
    nfsp0.save_checkpoint(CHECKPOINT_DIR, filename="nfsp_final.pt")
    print("\nPhase 2 complete. Final checkpoint saved.")

    # Save hand log
    log_path = os.path.join(RESULTS_DIR, "nfsp_hand_log.json")
    with open(log_path, "w") as f:
        json.dump(hand_log, f, indent=2)

    # Summary
    total = len(hand_log)
    wins  = sum(1 for h in hand_log if h["payoffs"][0] > 0)
    print(f"\n--- Summary ({total} total hands) ---")
    print(f"Overall win rate : {wins / total:.1%}")
    print(f"Final checkpoint : {CHECKPOINT_DIR}/nfsp_final.pt")
    print(f"Hand log         : {log_path}")


if __name__ == "__main__":
    run()
