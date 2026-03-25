from treys import Card, Evaluator

_evaluator = Evaluator()


def parse_cards(card_strs: list[str]) -> list[int]:
    """Convert string notation (e.g. 'Ah', 'Kd') to treys int format."""
    return [Card.new(c) for c in card_strs]


def evaluate(hole_cards: list[str], board: list[str]) -> int:
    """
    Returns a hand rank integer (lower = better).
    Requires at least 3 board cards (flop or later).
    """
    h = parse_cards(hole_cards)
    b = parse_cards(board)
    return _evaluator.evaluate(b, h)


def hand_strength_percentile(hole_cards: list[str], board: list[str]) -> float:
    """
    Returns hand strength as a percentile [0, 1] where 1.0 = best possible hand.
    Only valid postflop (board must have >= 3 cards).
    """
    rank = evaluate(hole_cards, board)
    # treys rank: 1 = royal flush, 7462 = worst hand
    return 1.0 - (rank / 7462.0)
