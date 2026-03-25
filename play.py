#!/usr/bin/env python3
"""
Play interactively against the trained NFSP bot.

Usage:
    python play.py
    python play.py --checkpoint results/nfsp_checkpoints/nfsp_final.pt
"""

import argparse
import os
import sys
import rlcard
from rlcard.agents import NFSPAgent
from rlcard.utils import set_seed

DEFAULT_CHECKPOINT = os.path.join("results", "nfsp_checkpoints", "nfsp_final.pt")

SUIT_SYMBOLS = {"H": "♥", "D": "♦", "C": "♣", "S": "♠"}

# Shortcut key → RLCard action name
ACTION_KEYS = {
    "F": "FOLD",
    "C": "CHECK_CALL",
    "H": "RAISE_HALF_POT",
    "P": "RAISE_POT",
    "A": "ALL_IN",
}


# ── display helpers ────────────────────────────────────────────────────────────

def fmt_card(c: str) -> str:
    """'HK' → 'K♥'"""
    suit = SUIT_SYMBOLS.get(c[0], c[0])
    rank = c[1]
    return f"{rank}{suit}"


def fmt_hand(cards: list) -> str:
    return "  ".join(fmt_card(c) for c in cards) if cards else "(none)"


# ── human agent ───────────────────────────────────────────────────────────────

class HumanAgent:
    def __init__(self, num_actions: int):
        self.use_raw = True
        self.num_actions = num_actions
        self.session_chips = 0   # set by main() before each hand

    def step(self, state: dict):
        return self._prompt(state)

    def eval_step(self, state: dict):
        return self._prompt(state), {}

    def _prompt(self, state: dict):
        raw     = state["raw_obs"]
        legal   = list(state["raw_legal_actions"])
        names   = {a.name: a for a in legal}

        all_chips = raw["all_chips"]
        my_chips  = raw["my_chips"]
        stakes    = raw.get("stakes", [])
        me        = raw.get("current_player", 0)

        pot     = int(raw.get("pot", sum(all_chips)))
        to_call = max(all_chips) - my_chips
        stakes  = raw.get("stakes", [])
        stack   = int(stakes[me]) if stakes else int(self.session_chips) - my_chips

        hand_str  = fmt_hand(raw['hand'])
        board_str = fmt_hand(raw.get('public_cards', []))
        pot_str   = f"Pot: {pot}   Stack: {stack}   To call: {to_call}"

        print()
        print("  ┌─────────────────────────────────────────┐")
        print(f"  │  Your hand : {hand_str}")
        print(f"  │  Board     : {board_str}")
        print(f"  │  {pot_str}")
        print("  ├─────────────────────────────────────────┤")
        for key, name in ACTION_KEYS.items():
            if name in names:
                print(f"  │    [{key}]  {name}")
        print("  └─────────────────────────────────────────┘")

        while True:
            raw_in = input("  Your action: ").strip().upper()
            if raw_in in ACTION_KEYS and ACTION_KEYS[raw_in] in names:
                chosen = names[ACTION_KEYS[raw_in]]
                print(f"  → You: {chosen.name}")
                return chosen
            if raw_in in names:
                print(f"  → You: {raw_in}")
                return names[raw_in]
            print(f"  Invalid — choose from: {[a.name for a in legal]}")


# ── bot setup ─────────────────────────────────────────────────────────────────

def make_nfsp(env) -> NFSPAgent:
    """Recreate the exact architecture used in train_nfsp.py."""
    return NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=[64, 64],
        reservoir_buffer_capacity=20000,
        anticipatory_param=0.1,
        batch_size=256,
        train_every=1,
        sl_learning_rate=5e-3,
        min_buffer_size_to_learn=100,
        q_replay_memory_size=20000,
        q_replay_memory_init_size=100,
        q_epsilon_start=0.06,
        q_epsilon_end=0.0,
        q_epsilon_decay_steps=1,     # inference only — decay irrelevant
        q_batch_size=32,
        q_train_every=1,
        q_mlp_layers=[64, 64],
        rl_learning_rate=5e-4,
        evaluate_with="average_policy",
    )


def load_bot(env, checkpoint_path: str) -> NFSPAgent:
    import torch
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Train the bot first with: python train_nfsp.py")
        sys.exit(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    nfsp = NFSPAgent.from_checkpoint(checkpoint)
    print(f"Loaded: {checkpoint_path}\n")
    return nfsp


# ── game loop ─────────────────────────────────────────────────────────────────

# Integer action IDs used by RLCard's NLHE environment
ACTION_INT_NAMES = {0: "FOLD", 1: "CHECK_CALL", 2: "RAISE_HALF_POT", 3: "RAISE_POT", 4: "ALL_IN"}


def _describe_bot_action(state: dict, action_int: int) -> str:
    raw     = state.get("raw_obs", {})
    all_chips = raw.get("all_chips", [0, 0])
    my_chips  = raw.get("my_chips", 0)
    pot       = int(raw.get("pot", sum(all_chips)))
    to_call   = max(all_chips) - my_chips
    name      = ACTION_INT_NAMES.get(action_int, str(action_int))

    if name == "FOLD":
        return "Bot folds"
    if name == "CHECK_CALL":
        return "Bot checks" if to_call == 0 else f"Bot calls {to_call}"
    if name == "RAISE_HALF_POT":
        return f"Bot raises {max(1, pot // 2)} chips (half pot)"
    if name == "RAISE_POT":
        return f"Bot raises {pot} chips (pot)"
    if name == "ALL_IN":
        stakes = raw.get("stakes", [])
        amount = int(stakes[1]) if len(stakes) > 1 else "?"
        return f"Bot goes all-in ({amount} chips)"
    return f"Bot: {name}"


def play_hand(env, human, nfsp) -> tuple:
    """Step through one hand manually. Returns (human_payoff, bot_hole_cards, final_board)."""
    state, player_id = env.reset()
    bot_hand   = []
    final_board = []

    while not env.is_over():
        raw = state.get("raw_obs", {})
        board = raw.get("public_cards", [])
        if board:
            final_board = board

        if player_id == 0:  # human
            action = human.eval_step(state)[0]
            state, player_id = env.step(action, raw_action=True)
        else:  # bot
            if not bot_hand:
                bot_hand = raw.get("hand", [])
            action, _ = nfsp.eval_step(state)
            # Never fold for free — override to check if there's nothing to call
            all_chips = raw.get("all_chips", [0, 0])
            to_call   = max(all_chips) - raw.get("my_chips", 0)
            if ACTION_INT_NAMES.get(action) == "FOLD" and to_call == 0:
                action = next(k for k, v in ACTION_INT_NAMES.items() if v == "CHECK_CALL")
            print(f"\n  → {_describe_bot_action(state, action)}")
            state, player_id = env.step(action, raw_action=False)

        # Capture board from the state returned after stepping —
        # critical for all-in runouts where RLCard deals remaining
        # streets without further player actions.
        post_board = state.get("raw_obs", {}).get("public_cards", [])
        if post_board:
            final_board = post_board

    return env.get_payoffs()[0], bot_hand, final_board


SESSION_CHIPS = 100   # starting chips per player


def make_env(stack: int):
    """Create a fresh env with the given starting stack for each player."""
    return rlcard.make(
        "no-limit-holdem",
        config={"num_players": 2, "allow_raw_data": True, "chips_for_each": stack},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--chips", type=int, default=SESSION_CHIPS,
                        help="Starting chip stack (default: 100)")
    args = parser.parse_args()

    set_seed(0)

    # Load bot using a temporary env just for its num_actions / state_shape
    tmp_env = make_env(args.chips)
    human = HumanAgent(num_actions=tmp_env.num_actions)
    nfsp  = load_bot(tmp_env, args.checkpoint)

    you = float(args.chips)
    bot = float(args.chips)
    total = 0

    print(f"  No-Limit Hold'em vs NFSP Bot")
    print(f"  Session starts: You {int(you)} — Bot {int(bot)}")
    print(f"  First to bust loses. Good luck.")
    print(f"  Actions: [F]old  [C]heck/Call  [H]alf-pot  [P]ot  [A]ll-in\n")

    while you > 0 and bot > 0:
        # Use the smaller stack as the effective stack so neither player
        # can bet more than they have in the session.
        effective = int(min(you, bot))
        env = make_env(effective)
        env.set_agents([human, nfsp])

        human.session_chips = you
        payoff, bot_hand, final_board = play_hand(env, human, nfsp)
        total += 1
        you += payoff
        bot -= payoff

        if payoff > 0:
            result = f"You won  +{int(payoff)} chips"
        elif payoff < 0:
            result = f"Bot won  +{int(-payoff)} chips"
        else:
            result = "Chop (tie)"

        print()
        print("  ┌─────────────────────────────────────────┐")
        print(f"  │  Board      : {fmt_hand(final_board) if final_board else '(folded preflop)'}")
        print(f"  │  Bot's hand : {fmt_hand(bot_hand)}")
        print(f"  │  {result}")
        print(f"  │  You: {int(you):<8}  Bot: {int(bot):<8}  Hand #{total}")
        print("  └─────────────────────────────────────────┘")

        if you <= 0:
            print("\n  You're out of chips. The bot wins the session.")
            break
        if bot <= 0:
            print("\n  Bot is out of chips. You win the session!")
            break

        # again = input("\n  Next hand? [y/n]: ").strip().lower()
        # if again != "y":
        #     print(f"\n  Session ended after {total} hands.")
        #     print(f"  You: {int(you)}  Bot: {int(bot)}")
        #     break

        print("===========================================")

if __name__ == "__main__":
    main()
