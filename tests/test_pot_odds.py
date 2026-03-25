import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pot_odds import pot_odds, expected_value, is_profitable_call, raise_size


# ---------------------------------------------------------------------------
# pot_odds()
# ---------------------------------------------------------------------------

def test_pot_odds_basic():
    assert abs(pot_odds(10, 30) - 0.25) < 0.001

def test_pot_odds_no_bet():
    assert pot_odds(0, 100) == 0.0

def test_pot_odds_half_pot():
    # Call 50 into 100: need 50/150 = 0.333
    assert abs(pot_odds(50, 100) - 1/3) < 0.001

def test_pot_odds_full_pot():
    # Call 100 into 100: need 100/200 = 0.50
    assert abs(pot_odds(100, 100) - 0.50) < 0.001

def test_pot_odds_overbet():
    # Call 200 into 100: need 200/300 = 0.666
    assert abs(pot_odds(200, 100) - 2/3) < 0.001

def test_pot_odds_min_bet():
    # Call 1 into 1000: need 1/1001 ≈ 0.001
    assert pot_odds(1, 1000) < 0.01

def test_pot_odds_always_less_than_one():
    for to_call, pot in [(5, 10), (100, 50), (1000, 1), (50, 50)]:
        assert pot_odds(to_call, pot) < 1.0

def test_pot_odds_always_positive():
    for to_call, pot in [(5, 10), (100, 50), (1, 1000)]:
        assert pot_odds(to_call, pot) > 0.0

def test_pot_odds_increases_with_bet_size():
    small = pot_odds(10, 100)
    large = pot_odds(80, 100)
    assert large > small


# ---------------------------------------------------------------------------
# expected_value()
# ---------------------------------------------------------------------------

def test_ev_break_even():
    # If equity exactly equals pot odds, EV should be ~0
    # pot=100, to_call=50, pot_odds=0.333, equity=0.333
    ev = expected_value(equity=1/3, pot=100, to_call=50)
    assert abs(ev) < 0.01

def test_ev_positive_when_equity_exceeds_pot_odds():
    ev = expected_value(equity=0.5, pot=100, to_call=10)
    assert ev > 0

def test_ev_negative_when_equity_below_pot_odds():
    ev = expected_value(equity=0.1, pot=100, to_call=50)
    assert ev < 0

def test_ev_coin_flip_half_pot():
    # 50% equity, calling 50 into 100: EV = 0.5*100 - 0.5*50 = 25
    ev = expected_value(equity=0.5, pot=100, to_call=50)
    assert abs(ev - 25.0) < 0.001

def test_ev_certain_winner():
    # 100% equity — always win the pot
    ev = expected_value(equity=1.0, pot=200, to_call=50)
    assert ev == 200.0 - 0.0

def test_ev_certain_loser():
    # 0% equity — always lose the call
    ev = expected_value(equity=0.0, pot=200, to_call=50)
    assert ev == -50.0

def test_ev_scales_with_pot():
    ev_small = expected_value(equity=0.6, pot=50, to_call=10)
    ev_large = expected_value(equity=0.6, pot=500, to_call=10)
    assert ev_large > ev_small


# ---------------------------------------------------------------------------
# is_profitable_call()
# ---------------------------------------------------------------------------

def test_profitable_call_basic():
    assert is_profitable_call(equity=0.40, to_call=10, pot=30)   # 0.40 > 0.25
    assert not is_profitable_call(equity=0.20, to_call=10, pot=30)  # 0.20 < 0.25

def test_profitable_call_at_break_even():
    # Exactly at pot odds — not profitable (not strictly greater)
    assert not is_profitable_call(equity=0.25, to_call=10, pot=30)

def test_profitable_call_large_overbet():
    # Opponent bets 3x pot — need 75% equity to call
    assert not is_profitable_call(equity=0.60, to_call=300, pot=100)
    assert is_profitable_call(equity=0.80, to_call=300, pot=100)

def test_profitable_call_tiny_bet():
    # 1-chip call into 1000 pot — almost any equity is profitable
    assert is_profitable_call(equity=0.01, to_call=1, pot=1000)

def test_not_profitable_zero_equity():
    assert not is_profitable_call(equity=0.0, to_call=10, pot=100)

def test_profitable_high_equity_any_bet():
    assert is_profitable_call(equity=0.99, to_call=500, pot=100)


# ---------------------------------------------------------------------------
# raise_size()
# ---------------------------------------------------------------------------

def test_raise_size_default_75pct():
    assert raise_size(100) == 75.0

def test_raise_size_half_pot():
    assert raise_size(100, fraction=0.5) == 50.0

def test_raise_size_full_pot():
    assert raise_size(100, fraction=1.0) == 100.0

def test_raise_size_overbet():
    assert raise_size(100, fraction=1.5) == 150.0

def test_raise_size_small_pot():
    assert raise_size(10, fraction=0.75) == 7.5

def test_raise_size_scales_with_pot():
    assert raise_size(200) == 2 * raise_size(100)

def test_raise_size_zero_pot():
    assert raise_size(0) == 0.0
