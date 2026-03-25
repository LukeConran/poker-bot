import random
from hand_evaluator import evaluate

SUITS = "cdhs"
RANKS = "23456789TJQKA"
FULL_DECK = [r + s for r in RANKS for s in SUITS]


def _random_board_completion(known_cards: list[str], target_size: int) -> list[str]:
    """Deal random cards to complete the board to target_size."""
    remaining = [c for c in FULL_DECK if c not in known_cards]
    needed = target_size - len([c for c in known_cards if c not in []])
    return random.sample(remaining, needed)


def monte_carlo_equity(
    hole_cards: list[str],
    board: list[str],
    num_opponents: int,
    simulations: int = 1000,
) -> float:
    """
    Estimate win equity via Monte Carlo simulation.
    Returns probability of winning [0, 1].
    """
    wins = 0
    ties = 0
    dead = hole_cards + board

    for _ in range(simulations):
        remaining = [c for c in FULL_DECK if c not in dead]
        random.shuffle(remaining)

        # Deal opponent hole cards
        opponents = []
        idx = 0
        for _ in range(num_opponents):
            opponents.append(remaining[idx:idx + 2])
            idx += 2

        # Complete the board to 5 cards
        sim_board = board + remaining[idx:idx + (5 - len(board))]

        our_rank = evaluate(hole_cards, sim_board)
        opp_ranks = [evaluate(opp, sim_board) for opp in opponents]
        best_opp = min(opp_ranks)

        if our_rank < best_opp:
            wins += 1
        elif our_rank == best_opp:
            ties += 1

    return (wins + ties * 0.5) / simulations
