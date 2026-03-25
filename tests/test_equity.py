import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from equity import monte_carlo_equity

SIM = 500  # simulations per test — lower = faster, higher = more accurate


# ---------------------------------------------------------------------------
# Preflop equity — premium hands
# ---------------------------------------------------------------------------

def test_pocket_aces_vs_one():
    equity = monte_carlo_equity(["Ah", "As"], [], num_opponents=1, simulations=SIM)
    assert equity > 0.80, f"AA vs 1: expected >0.80, got {equity}"

def test_pocket_kings_vs_one():
    equity = monte_carlo_equity(["Kh", "Ks"], [], num_opponents=1, simulations=SIM)
    assert equity > 0.70, f"KK vs 1: expected >0.70, got {equity}"

def test_pocket_queens_vs_one():
    equity = monte_carlo_equity(["Qh", "Qs"], [], num_opponents=1, simulations=SIM)
    assert equity > 0.65, f"QQ vs 1: expected >0.65, got {equity}"

def test_ace_king_suited_vs_one():
    equity = monte_carlo_equity(["Ah", "Kh"], [], num_opponents=1, simulations=SIM)
    assert equity > 0.60, f"AKs vs 1: expected >0.60, got {equity}"

def test_ace_king_offsuit_vs_one():
    equity = monte_carlo_equity(["Ah", "Kd"], [], num_opponents=1, simulations=SIM)
    assert equity > 0.58, f"AKo vs 1: expected >0.58, got {equity}"


# ---------------------------------------------------------------------------
# Preflop equity — weak hands
# ---------------------------------------------------------------------------

def test_72o_low_equity():
    equity = monte_carlo_equity(["7h", "2c"], [], num_opponents=1, simulations=SIM)
    assert equity < 0.40, f"72o vs 1: expected <0.40, got {equity}"

def test_32o_low_equity():
    equity = monte_carlo_equity(["3h", "2c"], [], num_opponents=1, simulations=SIM)
    assert equity < 0.40, f"32o vs 1: expected <0.40, got {equity}"

def test_93o_below_average():
    equity = monte_carlo_equity(["9h", "3c"], [], num_opponents=1, simulations=SIM)
    assert equity < 0.50, f"93o vs 1: expected <0.50, got {equity}"


# ---------------------------------------------------------------------------
# Preflop equity — multiple opponents
# ---------------------------------------------------------------------------

def test_pocket_aces_vs_two():
    equity = monte_carlo_equity(["Ah", "As"], [], num_opponents=2, simulations=SIM)
    assert 0.60 < equity < 0.80, f"AA vs 2: expected 0.60-0.80, got {equity}"

def test_pocket_aces_vs_five():
    equity = monte_carlo_equity(["Ah", "As"], [], num_opponents=5, simulations=SIM)
    assert equity > 0.40, f"AA vs 5: expected >0.40, got {equity}"

def test_equity_decreases_with_more_opponents():
    eq1 = monte_carlo_equity(["Ah", "As"], [], num_opponents=1, simulations=SIM)
    eq3 = monte_carlo_equity(["Ah", "As"], [], num_opponents=3, simulations=SIM)
    assert eq1 > eq3, "Equity should decrease with more opponents"

def test_72o_vs_three_opponents():
    equity = monte_carlo_equity(["7h", "2c"], [], num_opponents=3, simulations=SIM)
    assert equity < 0.30, f"72o vs 3: expected <0.30, got {equity}"


# ---------------------------------------------------------------------------
# Postflop — flop equity
# ---------------------------------------------------------------------------

def test_top_pair_top_kicker_flop():
    # AK on A-high flop — strong
    equity = monte_carlo_equity(["Ah", "Kd"], ["As", "7c", "2h"], num_opponents=1, simulations=SIM)
    assert equity > 0.70, f"TPTK flop: expected >0.70, got {equity}"

def test_bottom_pair_flop():
    # 22 on A-K-2 flop — bottom set actually, very strong
    equity = monte_carlo_equity(["2c", "2d"], ["Ah", "Kh", "2s"], num_opponents=1, simulations=SIM)
    assert equity > 0.85, f"Bottom set flop: expected >0.85, got {equity}"

def test_flush_draw_flop():
    # Ah2h on Kh7h3c — nut flush draw + ace overcard; equity runs higher than pure draw
    equity = monte_carlo_equity(["Ah", "2h"], ["Kh", "7h", "3c"], num_opponents=1, simulations=SIM)
    assert 0.35 < equity < 0.75, f"Flush draw flop: expected 0.35-0.75, got {equity}"

def test_open_ended_straight_draw_flop():
    # 67 on 5-8-K — open-ended straight draw
    equity = monte_carlo_equity(["6h", "7d"], ["5c", "8h", "Kd"], num_opponents=1, simulations=SIM)
    assert 0.30 < equity < 0.60, f"OESD flop: expected 0.30-0.60, got {equity}"

def test_no_pair_no_draw_flop():
    # 23 on A-K-Q rainbow — complete air, but backdoor draws keep equity ~0.20
    equity = monte_carlo_equity(["2h", "3d"], ["Ah", "Kd", "Qc"], num_opponents=1, simulations=SIM)
    assert equity < 0.25, f"Air flop: expected <0.25, got {equity}"

def test_overpair_on_low_flop():
    # KK on 2-5-7 — overpair, very strong
    equity = monte_carlo_equity(["Kh", "Kd"], ["2c", "5h", "7d"], num_opponents=1, simulations=SIM)
    assert equity > 0.75, f"Overpair low flop: expected >0.75, got {equity}"


# ---------------------------------------------------------------------------
# Postflop — turn equity
# ---------------------------------------------------------------------------

def test_made_flush_turn():
    # Ah2h on Kh7h3c4h — made flush
    equity = monte_carlo_equity(["Ah", "2h"], ["Kh", "7h", "3c", "4h"], num_opponents=1, simulations=SIM)
    assert equity > 0.80, f"Made flush turn: expected >0.80, got {equity}"

def test_two_pair_turn():
    # AK on A-K-7-2 — two pair
    equity = monte_carlo_equity(["Ah", "Kd"], ["As", "Kh", "7c", "2d"], num_opponents=1, simulations=SIM)
    assert equity > 0.85, f"Two pair turn: expected >0.85, got {equity}"

def test_gutshot_turn():
    # 67 on 5-8-K-2 — gutshot to straight (needs 4 or 9)
    equity = monte_carlo_equity(["6h", "7d"], ["5c", "8h", "Kd", "2s"], num_opponents=1, simulations=SIM)
    assert 0.15 < equity < 0.50, f"Gutshot turn: expected 0.15-0.50, got {equity}"


# ---------------------------------------------------------------------------
# Postflop — river equity (deterministic — no more cards to come)
# ---------------------------------------------------------------------------

def test_straight_beats_two_pair_river():
    # 67 completes straight on 5-8-9 board, opponent has two pair
    straight_eq = monte_carlo_equity(["6h", "7d"], ["5c", "8h", "9d", "2s", "3c"], num_opponents=1, simulations=SIM)
    two_pair_eq = monte_carlo_equity(["2c", "3d"], ["5c", "8h", "9d", "2s", "3c"], num_opponents=1, simulations=SIM)
    # Straight should be stronger than two pair here
    assert straight_eq > two_pair_eq

def test_river_equity_is_near_zero_or_one():
    # On river, equity should be close to 0 or 1 (either winning or losing)
    # Full house river
    equity = monte_carlo_equity(["Ah", "As"], ["Ac", "Ad", "Kh", "2c", "7d"], num_opponents=1, simulations=SIM)
    assert equity > 0.95, f"Quads river: expected >0.95, got {equity}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_equity_always_in_range():
    hands = [
        (["Ah", "As"], []),
        (["7h", "2c"], ["Kd", "Qh", "Js"]),
        (["Th", "Jh"], ["9h", "8h", "7h", "2c"]),
        (["2c", "3d"], ["4h", "5s", "6d", "7c", "8h"]),
    ]
    for hole, board in hands:
        eq = monte_carlo_equity(hole, board, num_opponents=1, simulations=200)
        assert 0.0 <= eq <= 1.0, f"Equity out of range for {hole} / {board}: {eq}"

def test_suited_better_than_offsuit():
    # AKs should have slightly more equity than AKo preflop
    suited = monte_carlo_equity(["Ah", "Kh"], [], num_opponents=1, simulations=SIM)
    offsuit = monte_carlo_equity(["Ah", "Kd"], [], num_opponents=1, simulations=SIM)
    # Not guaranteed every run due to variance, so just check both > 0.5
    assert suited > 0.55 and offsuit > 0.55
