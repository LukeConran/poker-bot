from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GameState:
    hole_cards: List[str]          # e.g. ["Ah", "Kd"]
    board: List[str]               # e.g. ["2c", "7h", "Ks"] — empty preflop
    pot: float                     # total chips in pot
    stack: float                   # our remaining stack
    to_call: float                 # amount we need to call (0 if no bet)
    position: str                  # "BTN", "SB", "BB", "EP", "MP", "CO"
    num_opponents: int             # number of active opponents

    @property
    def street(self) -> str:
        n = len(self.board)
        if n == 0:
            return "preflop"
        elif n == 3:
            return "flop"
        elif n == 4:
            return "turn"
        elif n == 5:
            return "river"
        else:
            raise ValueError(f"Invalid board length: {n}")

    @property
    def is_facing_bet(self) -> bool:
        return self.to_call > 0
