"""
Plot NFSP training progress from nfsp_hand_log.json.

Usage:
    python plot_training.py                          # uses default log path
    python plot_training.py results/nfsp_hand_log.json
"""

import json
import sys
import os

LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else "results/nfsp_hand_log.json"

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

if not os.path.exists(LOG_PATH):
    print(f"Log file not found: {LOG_PATH}")
    sys.exit(1)

with open(LOG_PATH) as f:
    log = json.load(f)

# Split into training payoffs and eval records
train_records = [e for e in log if "eval_vs_pokerbot" not in e]
eval_records  = [e for e in log if "eval_vs_pokerbot" in e]

# ── Figure layout ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("NFSP Training Progress", fontsize=14, fontweight="bold")

# ── 1. Rolling avg training payoff (self-play) ─────────────────────────────────
ax = axes[0][0]
if train_records:
    payoffs = [r["payoffs"][0] for r in train_records]
    window  = min(5000, len(payoffs) // 20 or 1)
    rolled  = [
        sum(payoffs[max(0, i - window):i]) / min(i, window)
        for i in range(1, len(payoffs) + 1)
    ]
    hands = [r["hand"] for r in train_records]
    ax.plot(hands, rolled, linewidth=0.8, color="steelblue")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_title(f"Self-play Payoff (rolling {window}-hand avg)")
    ax.set_xlabel("Hand (within phase)")
    ax.set_ylabel("Chips won / hand")
else:
    ax.text(0.5, 0.5, "No training data", transform=ax.transAxes, ha="center")
    ax.set_title("Self-play Payoff")

# ── 2. Eval payoff vs PokerBotAgent over time ──────────────────────────────────
ax = axes[0][1]
if eval_records:
    # Compute a cumulative hand count across phases for x-axis
    cum_hand = 0
    phase_offsets = {}
    prev_phase, prev_hand = None, 0
    xs, ys = [], []
    for r in eval_records:
        if r["phase"] != prev_phase:
            cum_hand += prev_hand
            phase_offsets[r["phase"]] = cum_hand
            prev_phase = r["phase"]
        xs.append(phase_offsets[r["phase"]] + r["hand"])
        ys.append(r["eval_vs_pokerbot"])
        prev_hand = r["hand"]

    ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.2, color="darkorange", label="eval")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    # shade positive region
    ax.fill_between(xs, ys, 0, where=[y > 0 for y in ys], alpha=0.2, color="green", label="+EV")
    ax.fill_between(xs, ys, 0, where=[y < 0 for y in ys], alpha=0.2, color="red",   label="-EV")
    ax.set_title(f"Eval Payoff vs PokerBot ({len(eval_records)} checkpoints)")
    ax.set_xlabel("Total hands trained")
    ax.set_ylabel("Avg chips / hand (1k hands)")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No eval data yet.\nEval points are saved every EVAL_EVERY hands.",
            transform=ax.transAxes, ha="center", va="center", fontsize=10)
    ax.set_title("Eval Payoff vs PokerBot")

# ── 3. Win rate (rolling) ──────────────────────────────────────────────────────
ax = axes[1][0]
if train_records:
    wins   = [1 if r["payoffs"][0] > 0 else 0 for r in train_records]
    window = min(5000, len(wins) // 20 or 1)
    rolled = [
        sum(wins[max(0, i - window):i]) / min(i, window)
        for i in range(1, len(wins) + 1)
    ]
    ax.plot(hands, rolled, linewidth=0.8, color="mediumseagreen")
    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", label="50% baseline")
    ax.set_title(f"Win Rate (rolling {window}-hand avg)")
    ax.set_xlabel("Hand (within phase)")
    ax.set_ylabel("Win rate")
    ax.set_ylim(0.3, 0.7)
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No training data", transform=ax.transAxes, ha="center")
    ax.set_title("Win Rate")

# ── 4. Payoff distribution histogram ──────────────────────────────────────────
ax = axes[1][1]
if train_records:
    payoffs = [r["payoffs"][0] for r in train_records]
    ax.hist(payoffs, bins=40, color="slateblue", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", linewidth=1, linestyle="--")
    mean_p = sum(payoffs) / len(payoffs)
    ax.axvline(mean_p, color="orange", linewidth=1.2, label=f"mean={mean_p:+.3f}")
    ax.set_title("Training Payoff Distribution")
    ax.set_xlabel("Chips won / hand")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No training data", transform=ax.transAxes, ha="center")
    ax.set_title("Payoff Distribution")

plt.tight_layout()
out_path = LOG_PATH.replace(".json", "_plot.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.show()
