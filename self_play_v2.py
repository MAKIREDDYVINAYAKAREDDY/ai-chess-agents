import os
import shutil
import chess
import numpy as np
from stable_baselines3 import PPO

from chess_env_v4 import ChessEnvV4


BASE_MODEL = "chess_ai_agent_v2.zip"
OUTPUT_MODEL = "chess_ai_agent_v2_selfplay"

SELF_PLAY_GAMES = 200
MAX_MOVES = 200


def find_stockfish():
    paths = [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/games/stockfish",
    ]

    for path in paths:
        if os.path.exists(path):
            return path

    return shutil.which("stockfish")


def get_v2_move(model, env):
    legal_actions = env.legal_actions()

    if len(legal_actions) == 0:
        return None

    observation = env._get_observation()

    action, _ = model.predict(
        observation,
        deterministic=False
    )

    action = int(action)

    # Safety fallback because standard SB3 PPO
    # does not directly consume our action mask.
    if action not in legal_actions:
        action = int(
            np.random.choice(legal_actions)
        )

    return env.action_to_move(action)


def create_game(model_white, model_black):
    env = ChessEnvV2 = ChessEnvV4(
        max_moves=MAX_MOVES
    )

    env.reset()

    board = env.board

    positions = []

    while (
        not board.is_game_over()
        and env.move_count < MAX_MOVES
    ):

        # Save current position.
        positions.append(
            board.fen()
        )

        if board.turn == chess.WHITE:
            model = model_white
        else:
            model = model_black

        move = get_v2_move(
            model,
            env
        )

        if move is None:
            break

        board.push(move)

        env.move_count += 1

    result = board.result(
        claim_draw=True
    )

    env.close()

    return result, env.move_count


def main():

    print("=" * 70)
    print("AI CHESS AGENT - V2.7 SELF PLAY")
    print("=" * 70)

    print("\nLoading V2 model...")

    model = PPO.load(
        BASE_MODEL
    )

    print(
        "Loaded:",
        BASE_MODEL
    )

    print(
        "\nSelf-play games:",
        SELF_PLAY_GAMES
    )

    wins_white = 0
    wins_black = 0
    draws = 0

    total_moves = 0

    for game in range(
        1,
        SELF_PLAY_GAMES + 1
    ):

        # Same model controls both sides.
        result, moves = create_game(
            model,
            model
        )

        total_moves += moves

        if result == "1-0":
            wins_white += 1

        elif result == "0-1":
            wins_black += 1

        else:
            draws += 1

        if game % 10 == 0:

            completed = (
                wins_white
                + wins_black
                + draws
            )

            print(
                f"Game {game:03d}/{SELF_PLAY_GAMES} | "
                f"White wins: {wins_white} | "
                f"Black wins: {wins_black} | "
                f"Draws: {draws} | "
                f"Avg moves: "
                f"{total_moves / completed:.1f}"
            )

    print("\n" + "=" * 70)
    print("SELF-PLAY EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "\nWhite wins:",
        wins_white
    )

    print(
        "Black wins:",
        wins_black
    )

    print(
        "Draws:",
        draws
    )

    print(
        "Average moves:",
        f"{total_moves / SELF_PLAY_GAMES:.1f}"
    )

    # --------------------------------------------------------
    # Save a copy of the current model.
    #
    # We deliberately don't overwrite the baseline.
    # --------------------------------------------------------

    model.save(
        OUTPUT_MODEL
    )

    print(
        "\nSaved:",
        OUTPUT_MODEL + ".zip"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This phase validates self-play infrastructure."
    )

    print(
        "The model has NOT been gradient-trained during"
        " these games."
    )

    print(
        "Actual self-play learning comes in the next phase."
    )


if __name__ == "__main__":
    main()
