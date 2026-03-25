import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hand_evaluator import evaluate, hand_strength_percentile


# ---------------------------------------------------------------------------
# Hand rankings — stronger hand should have lower rank (treys convention)
# ---------------------------------------------------------------------------

def test_straight_flush_beats_four_of_a_kind():
    sf = evaluate(["9h", "8h"], ["7h", "6h", "5h", "2c", "Kd"])
    quads = evaluate(["Ah", "As"], ["Ac", "Ad", "Kh", "2c", "3d"])
    assert sf < quads

def test_four_of_a_kind_beats_full_house():
    quads = evaluate(["Ah", "As"], ["Ac", "Ad", "Kh", "2c", "3d"])
    boat = evaluate(["Ah", "As"], ["Ac", "Kh", "Kd", "2c", "3d"])
    assert quads < boat

def test_full_house_beats_flush():
    boat = evaluate(["Ah", "As"], ["Ac", "Kh", "Kd", "2c", "3d"])
    flush = evaluate(["Ah", "Kh"], ["2h", "7h", "Th", "3c", "9d"])
    assert boat < flush

def test_flush_beats_straight():
    flush = evaluate(["Ah", "Kh"], ["2h", "7h", "Th", "3c", "9d"])
    straight = evaluate(["6h", "7d"], ["5c", "8h", "9d", "2s", "3c"])
    assert flush < straight

def test_straight_beats_three_of_a_kind():
    straight = evaluate(["6h", "7d"], ["5c", "8h", "9d", "2s", "3c"])
    trips = evaluate(["Ah", "As"], ["Ac", "Kh", "Qd", "2c", "3d"])
    assert straight < trips

def test_three_of_a_kind_beats_two_pair():
    trips = evaluate(["Ah", "As"], ["Ac", "Kh", "Qd", "2c", "3d"])
    two_pair = evaluate(["Ah", "Kd"], ["As", "Kh", "Qd", "2c", "3d"])
    assert trips < two_pair

def test_two_pair_beats_one_pair():
    two_pair = evaluate(["Ah", "Kd"], ["As", "Kh", "Qd", "2c", "3d"])
    one_pair = evaluate(["Ah", "2d"], ["As", "Kh", "Qd", "Jc", "3d"])
    assert two_pair < one_pair

def test_one_pair_beats_high_card():
    one_pair = evaluate(["Ah", "2d"], ["As", "Kh", "Qd", "Jc", "3d"])
    high_card = evaluate(["Ah", "2d"], ["7s", "Kh", "Qd", "Jc", "3d"])
    assert one_pair < high_card


# ---------------------------------------------------------------------------
# Kicker matters
# ---------------------------------------------------------------------------

def test_higher_kicker_wins():
    # Both have pair of aces, different kickers
    ace_king = evaluate(["Ah", "Kd"], ["As", "2h", "3c", "4d", "5s"])
    ace_queen = evaluate(["Ah", "Qd"], ["As", "2h", "3c", "4d", "5s"])
    # Note: both have A-2-3-4-5 straight here — let's use a cleaner board
    ace_king2 = evaluate(["Ah", "Kd"], ["As", "7h", "3c", "4d", "9s"])
    ace_queen2 = evaluate(["Ah", "Qd"], ["As", "7h", "3c", "4d", "9s"])
    assert ace_king2 < ace_queen2  # AK beats AQ

def test_higher_pair_wins():
    kings = evaluate(["Kh", "Kd"], ["As", "7h", "3c", "4d", "9s"])
    queens = evaluate(["Qh", "Qd"], ["As", "7h", "3c", "4d", "9s"])
    assert kings < queens


# ---------------------------------------------------------------------------
# Board plays (best 5 of 7)
# ---------------------------------------------------------------------------

def test_board_straight_both_play():
    # Both players use the board straight, split pot — equal rank
    board = ["5c", "6h", "7d", "8s", "9c"]
    player1 = evaluate(["Ah", "Kd"], board)
    player2 = evaluate(["2h", "3d"], board)
    assert player1 == player2  # Both play the board

def test_player_improves_board():
    # Board has 4 to a flush; player completes it
    with_flush = evaluate(["Ah", "Kh"], ["2h", "7h", "Th", "3c", "9d"])
    without_flush = evaluate(["2c", "3d"], ["2h", "7h", "Th", "3c", "9d"])
    assert with_flush < without_flush


# ---------------------------------------------------------------------------
# Specific hand types
# ---------------------------------------------------------------------------

def test_royal_flush():
    rank = evaluate(["Ah", "Kh"], ["Qh", "Jh", "Th", "2c", "3d"])
    assert rank == 1  # treys: rank 1 = royal flush

def test_straight_flush():
    rank = evaluate(["9h", "8h"], ["7h", "6h", "5h", "2c", "Kd"])
    assert rank < 10  # straight flushes are ranks 1-10

def test_wheel_straight():
    # A-2-3-4-5 straight (ace plays low)
    rank = evaluate(["Ah", "2d"], ["3c", "4h", "5s", "9d", "Kc"])
    straight_rank = evaluate(["6h", "7d"], ["5c", "8h", "9d", "2s", "3c"])
    # Both should be straights (rank < trips threshold)
    trips = evaluate(["Ah", "As"], ["Ac", "Kh", "Qd", "2c", "3d"])
    assert rank < trips
    assert straight_rank < trips

def test_broadway_straight():
    # A-K-Q-J-T straight
    rank = evaluate(["Ah", "Kd"], ["Qc", "Jh", "Ts", "2d", "3c"])
    trips = evaluate(["Ah", "As"], ["Ac", "Kh", "Qd", "2c", "3d"])
    assert rank < trips


# ---------------------------------------------------------------------------
# hand_strength_percentile()
# ---------------------------------------------------------------------------

def test_percentile_always_in_range():
    hands = [
        (["Ah", "As"], ["Ac", "Ad", "Kh"]),
        (["7h", "2c"], ["Kd", "Qh", "Js"]),
        (["Th", "Jh"], ["9h", "8h", "7h"]),
        (["2c", "3d"], ["4h", "5s", "6d"]),
        (["Kh", "Kd"], ["Kc", "2h", "3d"]),
    ]
    for hole, board in hands:
        p = hand_strength_percentile(hole, board)
        assert 0.0 <= p <= 1.0, f"Percentile out of range for {hole}/{board}: {p}"

def test_quads_near_top():
    p = hand_strength_percentile(["Ah", "As"], ["Ac", "Ad", "Kh"])
    assert p > 0.99

def test_straight_flush_near_top():
    p = hand_strength_percentile(["9h", "8h"], ["7h", "6h", "5h"])
    assert p > 0.99

def test_high_card_low_percentile():
    p = hand_strength_percentile(["2c", "3d"], ["Ah", "Kh", "Qd"])
    assert p < 0.20

def test_stronger_hand_higher_percentile():
    flush = hand_strength_percentile(["Ah", "Kh"], ["2h", "7h", "Th"])
    pair = hand_strength_percentile(["2c", "2d"], ["Ah", "Kh", "Th"])
    assert flush > pair

def test_full_house_high_percentile():
    p = hand_strength_percentile(["Ah", "As"], ["Ac", "Kh", "Kd"])
    assert p > 0.95

def test_two_pair_mid_percentile():
    p = hand_strength_percentile(["Ah", "Kd"], ["As", "Kh", "2c"])
    assert 0.60 < p < 0.95

def test_one_pair_aces_mid_percentile():
    p = hand_strength_percentile(["Ah", "2d"], ["As", "7h", "3c"])
    assert 0.40 < p < 0.80
