from game_state import GameState
from equity import monte_carlo_equity
from pot_odds import is_profitable_call, expected_value, raise_size

# Thresholds
RAISE_EQUITY_THRESHOLD = 0.65   # raise if equity >= this
CALL_EQUITY_THRESHOLD = 0.35    # call if equity >= pot odds (handled by pot_odds), but don't call below this
BLUFF_FREQUENCY = 0.1           # probability of bluffing when weak


class Bot:
    def decide(self, state: GameState) -> dict:
        """
        Returns a decision dict: {"action": "fold"|"call"|"raise", "amount": float|None}
        """
        equity = monte_carlo_equity(
            state.hole_cards,
            state.board,
            state.num_opponents,
        )

        if not state.is_facing_bet:
            return self._acting_first(state, equity)
        else:
            return self._facing_bet(state, equity)

    def _acting_first(self, state: GameState, equity: float) -> dict:
        """No bet to face — choose between check and bet."""
        if equity >= RAISE_EQUITY_THRESHOLD:
            amount = raise_size(state.pot)
            return {"action": "raise", "amount": amount}

        import random
        if random.random() < BLUFF_FREQUENCY:
            amount = raise_size(state.pot, fraction=0.5)
            return {"action": "raise", "amount": amount}

        return {"action": "check", "amount": None}

    def _facing_bet(self, state: GameState, equity: float) -> dict:
        """Facing a bet — choose between fold, call, and raise."""
        ev = expected_value(equity, state.pot, state.to_call)

        if equity >= RAISE_EQUITY_THRESHOLD:
            amount = raise_size(state.pot)
            return {"action": "raise", "amount": amount}

        if is_profitable_call(equity, state.to_call, state.pot) and equity >= CALL_EQUITY_THRESHOLD:
            return {"action": "call", "amount": state.to_call}

        return {"action": "fold", "amount": None}
