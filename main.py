from game_state import GameState
from bot import Bot


def main():
    # Example: heads-up hand setup
    state = GameState(
        hole_cards=["Ah", "Kd"],
        board=[],
        pot=10,
        stack=200,
        to_call=5,
        position="BTN",
        num_opponents=1,
    )

    bot = Bot()
    action = bot.decide(state)
    print(f"Bot action: {action}")


if __name__ == "__main__":
    main()
