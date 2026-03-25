def pot_odds(to_call: float, pot: float) -> float:
    """
    Returns the pot odds as a required equity fraction to break even.
    e.g. to_call=10, pot=30 -> need 10/(30+10) = 0.25 equity to call.
    """
    if to_call <= 0:
        return 0.0
    return to_call / (pot + to_call)


def expected_value(equity: float, pot: float, to_call: float) -> float:
    """
    Simple EV of a call: (equity * pot_won) - (1 - equity) * to_call
    """
    return equity * pot - (1 - equity) * to_call


def is_profitable_call(equity: float, to_call: float, pot: float) -> bool:
    return equity > pot_odds(to_call, pot)


def raise_size(pot: float, fraction: float = 0.75) -> float:
    """Returns a raise size as a fraction of the pot."""
    return round(pot * fraction, 2)
