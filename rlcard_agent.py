from game_state import GameState
from bot import Bot


class PokerBotAgent:
    """
    Wraps our probability-based Bot in RLCard's agent interface.
    Uses raw observations so we can access hole cards, board, and chip counts.
    RLCard duck-types agents — only step(), eval_step(), and use_raw are required.
    """

    def __init__(self):
        self.use_raw = True  # receive raw state dicts instead of encoded arrays
        self.bot = Bot()

    def step(self, state: dict):
        """Called during training — returns an Action enum value."""
        return self._decide(state)

    def eval_step(self, state: dict) -> tuple:
        """Called during evaluation — deterministic, returns (action, info)."""
        return self._decide(state), {}

    @staticmethod
    def _normalize_cards(cards: list[str]) -> list[str]:
        """
        Convert RLCard suit-first format ('HK', 'D4') to treys format ('Kh', '4d').
        RLCard: suit char + rank char  e.g. 'HK' = King of Hearts
        treys:  rank char + suit char  e.g. 'Kh' = King of Hearts
        """
        return [c[1] + c[0].lower() for c in cards]

    def _decide(self, state: dict):
        raw = state["raw_obs"]
        legal = list(state["raw_legal_actions"])  # list of Action enum values

        hole_cards = self._normalize_cards(raw["hand"])
        board = self._normalize_cards(raw.get("public_cards", []))

        all_chips = raw["all_chips"]   # chips committed by each player this round
        my_chips = raw["my_chips"]     # our committed chips
        stakes = raw.get("stakes", []) # remaining stacks per player

        pot = int(raw.get("pot", sum(all_chips)))
        to_call = max(all_chips) - my_chips
        current_player = raw.get("current_player", 0)
        stack = stakes[current_player] if stakes else 100

        game_state = GameState(
            hole_cards=hole_cards,
            board=board,
            pot=max(pot, 1),
            stack=stack,
            to_call=to_call,
            position="BTN",
            num_opponents=len(all_chips) - 1,
        )

        decision = self.bot.decide(game_state)
        return self._map_action(decision["action"], legal)

    @staticmethod
    def _map_action(action: str, legal: list):
        """
        Map fold/check/call/raise to the best available RLCard Action enum value.
        Legal actions are Action enum objects; match by .name.
        """
        names = {a.name: a for a in legal}
        # Action enum names in RLCard: FOLD, CHECK_CALL, RAISE_HALF_POT, RAISE_POT, ALL_IN

        if action == "fold":
            if "FOLD" in names:
                return names["FOLD"]

        if action in ("check", "call"):
            if "CHECK_CALL" in names:
                return names["CHECK_CALL"]

        if action == "raise":
            for preferred in ("RAISE_POT", "RAISE_HALF_POT", "ALL_IN", "CHECK_CALL"):
                if preferred in names:
                    return names[preferred]

        # Fallback: prefer CHECK_CALL over FOLD
        for preferred in ("CHECK_CALL", "RAISE_HALF_POT", "FOLD"):
            if preferred in names:
                return names[preferred]

        return legal[0]
