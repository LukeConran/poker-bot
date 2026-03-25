"""
Self-play training loop using RLCard's No-Limit Hold'em environment.

Runs N hands between two instances of our PokerBotAgent, logs results,
and prints a summary. This generates the game data needed for future
learning (trajectories are stored in `results/`).
"""

import json
import os
import time
import rlcard
from rlcard.utils import set_seed

from rlcard_agent import PokerBotAgent


RESULTS_DIR = "results"
NUM_HANDS = 400
EVAL_EVERY = 200   # print running stats every N hands
SEED = 42


def run(num_hands: int = NUM_HANDS):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    env = rlcard.make(
        "no-limit-holdem",
        config={
            "seed": SEED,
            "allow_raw_data": True,
            "num_players": 2,
        },
    )

    agents = [PokerBotAgent(), PokerBotAgent()]
    env.set_agents(agents)

    all_payoffs = []
    hand_log = []
    start = time.time()

    for hand_num in range(1, num_hands + 1):
        trajectories, payoffs = env.run(is_training=False)

        all_payoffs.append(payoffs)
        hand_log.append({
            "hand": hand_num,
            "payoffs": list(payoffs),
            "steps": [len(t) for t in trajectories],
        })

        if hand_num % EVAL_EVERY == 0:
            recent = all_payoffs[-EVAL_EVERY:]
            p0_avg = sum(p[0] for p in recent) / EVAL_EVERY
            elapsed = time.time() - start
            print(
                f"Hand {hand_num:>5} | "
                f"P0 avg payoff (last {EVAL_EVERY}): {p0_avg:+.2f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    # Save full hand log (convert numpy types to native Python for JSON)
    log_path = os.path.join(RESULTS_DIR, "hand_log.json")
    with open(log_path, "w") as f:
        json.dump(hand_log, f, default=lambda x: int(x) if hasattr(x, '__index__') else float(x))

    # Summary
    p0_total = sum(p[0] for p in all_payoffs)
    p0_wins = sum(1 for p in all_payoffs if p[0] > 0)
    print(f"\n--- Summary ({num_hands} hands) ---")
    print(f"Player 0 total payoff : {p0_total:+.1f}")
    print(f"Player 0 win rate     : {p0_wins / num_hands:.1%}")
    print(f"Results saved to      : {log_path}")


if __name__ == "__main__":
    run()
