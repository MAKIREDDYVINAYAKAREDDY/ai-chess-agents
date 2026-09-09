import os
import chess
import chess.engine
from stable_baselines3 import PPO

from chess_env_v4 import ChessEnvV4


V1_MODEL = "chess_ai_agent.h5"
V2_MODEL = "chess_ai_agent_v2.zip"

STOCKFISH_PATHS = [
    "/opt/homebrew/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/games/stockfish",
]


def find_stockfish():
    for path in STOCKFISH_PATHS:
        if os.path.exists(path):
            return path

    import shutil
    path = shutil.which("stockfish")

    if path:
        return path

    raise FileNotFoundError(
        "Stockfish executable not found."
    )


def state_for_model(env):
    return env._get_observation()


def choose_v2_move(model, env):
    legal_actions = env.legal_actions()

    if len(legal_actions) == 0:
        return None

    obs = state_for_model(env)

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    action = int(action)

    # Safety fallback:
    # PPO may select an illegal action because standard
    # SB3 PPO does not natively apply our legal-action mask.
    if action not in legal_actions:
        action = int(legal_actions[0])

    return env.action_to_move(action)


def choose_v1_move(model, board):
    """
    Compatibility evaluation of the original V1 model.

    The V1 model expects the original 8x8x12 observation
    and its original modulo legal-move mapping.
    """

    state = legacy_state(board)

    action, _ = model.predict(
        state,
        deterministic=True
    )

    action = int(action)

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    return legal_moves[
        action % len(legal_moves)
    ]


def legacy_state(board):
    import numpy as np

    state = np.zeros(
        (8, 8, 12),
        dtype=np.uint8
    )

    for square, piece in board.piece_map().items():

        row, col = divmod(
            square,
            8
        )

        piece_idx = (
            piece.piece_type - 1
        )

        if piece.color == chess.BLACK:
            piece_idx += 6

        state[
            row,
            col,
            piece_idx
        ] = 1

    return state


def stockfish_move(board, engine, level=5):

    result = engine.play(
        board,
        chess.engine.Limit(
            depth=level
        )
    )

    return result.move


def play_v2_vs_stockfish(
    model,
    engine,
    v2_white=True,
    stockfish_depth=5
):

    env = ChessEnvV4(
        max_moves=200
    )

    env.reset()

    board = env.board

    moves = 0

    while not board.is_game_over() and moves < 200:

        if (
            board.turn == chess.WHITE
            and v2_white
        ) or (
            board.turn == chess.BLACK
            and not v2_white
        ):

            move = choose_v2_move(
                model,
                env
            )

        else:

            move = stockfish_move(
                board,
                engine,
                stockfish_depth
            )

        if move is None:
            break

        board.push(move)
        moves += 1

    result = board.result(
        claim_draw=True
    )

    env.close()

    return result, moves


def play_v1_vs_stockfish(
    model,
    engine,
    v1_white=True,
    stockfish_depth=5
):

    board = chess.Board()

    moves = 0

    while not board.is_game_over() and moves < 200:

        if (
            board.turn == chess.WHITE
            and v1_white
        ) or (
            board.turn == chess.BLACK
            and not v1_white
        ):

            move = choose_v1_move(
                model,
                board
            )

        else:

            move = stockfish_move(
                board,
                engine,
                stockfish_depth
            )

        if move is None:
            break

        board.push(move)
        moves += 1

    return (
        board.result(claim_draw=True),
        moves
    )


def calculate_elo(win_rate, draw_rate):

    # Expected score:
    # win = 1
    # draw = 0.5
    # loss = 0

    score = (
        win_rate
        + 0.5 * draw_rate
    )

    score = max(
        0.01,
        min(0.99, score)
    )

    import math

    return (
        400
        * math.log10(
            score / (1 - score)
        )
    )


def summarize(
    name,
    results
):

    wins = results.count("win")
    draws = results.count("draw")
    losses = results.count("loss")

    total = len(results)

    win_rate = wins / total
    draw_rate = draws / total
    loss_rate = losses / total

    score = (
        win_rate
        + 0.5 * draw_rate
    )

    elo_diff = calculate_elo(
        win_rate,
        draw_rate
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Games:", total)
    print("Wins:", wins)
    print("Draws:", draws)
    print("Losses:", losses)

    print(
        f"Win rate:  {win_rate:.1%}"
    )

    print(
        f"Draw rate: {draw_rate:.1%}"
    )

    print(
        f"Loss rate: {loss_rate:.1%}"
    )

    print(
        f"Score:     {score:.3f}"
    )

    print(
        f"Elo diff vs Stockfish: "
        f"{elo_diff:+.0f}"
    )


def main():

    print("=" * 70)
    print("AI CHESS AGENT - V2.6 EVALUATION")
    print("=" * 70)

    print("\nLoading models...")

    v1 = PPO.load(
        V1_MODEL
    )

    v2 = PPO.load(
        V2_MODEL
    )

    stockfish_path = find_stockfish()

    print(
        "Stockfish:",
        stockfish_path
    )

    engine = chess.engine.SimpleEngine.popen_uci(
        stockfish_path
    )

    engine.configure(
        {
            "Skill Level": 5
        }
    )

    games = 10

    # --------------------------------------------------------
    # V2 evaluation
    # --------------------------------------------------------

    v2_results = []
    v2_lengths = []

    print("\n")
    print("=" * 70)
    print("V2 PPO vs STOCKFISH")
    print("=" * 70)

    for i in range(games):

        v2_white = (
            i % 2 == 0
        )

        result, length = (
            play_v2_vs_stockfish(
                v2,
                engine,
                v2_white=v2_white
            )
        )

        if (
            result == "1-0"
            and v2_white
        ) or (
            result == "0-1"
            and not v2_white
        ):
            outcome = "win"

        elif result == "1/2-1/2":
            outcome = "draw"

        else:
            outcome = "loss"

        v2_results.append(
            outcome
        )

        v2_lengths.append(
            length
        )

        print(
            f"Game {i + 1:02d}: "
            f"{outcome.upper():5s} | "
            f"moves={length}"
        )

    summarize(
        "V2 PPO",
        v2_results
    )

    print(
        "Average game length:",
        f"{sum(v2_lengths) / len(v2_lengths):.1f}"
    )

    # --------------------------------------------------------
    # V1 evaluation
    # --------------------------------------------------------

    v1_results = []
    v1_lengths = []

    print("\n")
    print("=" * 70)
    print("V1 PPO vs STOCKFISH")
    print("=" * 70)

    for i in range(games):

        v1_white = (
            i % 2 == 0
        )

        result, length = (
            play_v1_vs_stockfish(
                v1,
                engine,
                v1_white=v1_white
            )
        )

        if (
            result == "1-0"
            and v1_white
        ) or (
            result == "0-1"
            and not v1_white
        ):
            outcome = "win"

        elif result == "1/2-1/2":
            outcome = "draw"

        else:
            outcome = "loss"

        v1_results.append(
            outcome
        )

        v1_lengths.append(
            length
        )

        print(
            f"Game {i + 1:02d}: "
            f"{outcome.upper():5s} | "
            f"moves={length}"
        )

    summarize(
        "V1 PPO",
        v1_results
    )

    print(
        "Average game length:",
        f"{sum(v1_lengths) / len(v1_lengths):.1f}"
    )

    engine.quit()

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
