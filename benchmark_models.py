import os
import math
import chess
import chess.engine
import numpy as np

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO


V1_MODEL = "chess_ai_agent.h5"
V2_MODEL = "chess_ai_agent_v2.zip"
MASKED_MODEL = "chess_ai_agent_v2_masked.zip"

GAMES = 10
STOCKFISH_DEPTH = 5
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

    import shutil
    path = shutil.which("stockfish")

    if path:
        return path

    raise FileNotFoundError(
        "Stockfish executable not found."
    )


def legacy_state(board):
    state = np.zeros(
        (8, 8, 12),
        dtype=np.uint8
    )

    for square, piece in board.piece_map().items():

        row, col = divmod(square, 8)

        piece_index = piece.piece_type - 1

        if piece.color == chess.BLACK:
            piece_index += 6

        state[row, col, piece_index] = 1

    return state


def v1_move(model, board):
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    state = legacy_state(board)

    action, _ = model.predict(
        state,
        deterministic=True
    )

    action = int(action)

    return legal_moves[
        action % len(legal_moves)
    ]


def v2_state(board):
    state = np.zeros(
        (8, 8, 18),
        dtype=np.float32
    )

    for square, piece in board.piece_map().items():

        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        piece_index = piece.piece_type - 1

        if piece.color == chess.BLACK:
            piece_index += 6

        state[
            row,
            col,
            piece_index
        ] = 1.0

    if board.turn == chess.WHITE:
        state[:, :, 12] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        state[:, :, 13] = 1.0

    if board.has_queenside_castling_rights(chess.WHITE):
        state[:, :, 14] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK):
        state[:, :, 15] = 1.0

    if board.has_queenside_castling_rights(chess.BLACK):
        state[:, :, 16] = 1.0

    if board.ep_square is not None:

        row = 7 - chess.square_rank(
            board.ep_square
        )

        col = chess.square_file(
            board.ep_square
        )

        state[row, col, 17] = 1.0

    return state.flatten()


def move_to_action(move):

    base = (
        move.from_square * 64
        + move.to_square
    )

    if move.promotion is None:
        return base

    promotion = {
        chess.QUEEN: 0,
        chess.ROOK: 1,
        chess.BISHOP: 2,
        chess.KNIGHT: 3,
    }[move.promotion]

    return 4096 + base * 4 + promotion


def action_to_move(action):

    action = int(action)

    if action < 4096:

        return chess.Move(
            action // 64,
            action % 64
        )

    action -= 4096

    promotion = {
        0: chess.QUEEN,
        1: chess.ROOK,
        2: chess.BISHOP,
        3: chess.KNIGHT,
    }[action % 4]

    base = action // 4

    return chess.Move(
        base // 64,
        base % 64,
        promotion=promotion
    )


def v2_move(model, board):

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    state = v2_state(board)

    action, _ = model.predict(
        state,
        deterministic=True
    )

    action = int(action)

    move = action_to_move(action)

    if move in legal_moves:
        return move

    # Safe fallback.
    return legal_moves[0]


def masked_move(model, board):

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    state = v2_state(board)

    legal_actions = np.array(
        [
            move_to_action(move)
            for move in legal_moves
        ],
        dtype=np.int64
    )

    action, _ = model.predict(
        state,
        deterministic=True,
        action_masks=np.isin(
            np.arange(20480),
            legal_actions
        )
    )

    return action_to_move(
        int(action)
    )


def stockfish_move(board, engine):

    result = engine.play(
        board,
        chess.engine.Limit(
            depth=STOCKFISH_DEPTH
        )
    )

    return result.move


def play_game(
    model,
    model_type,
    engine,
    model_white
):

    board = chess.Board()

    move_count = 0

    while (
        not board.is_game_over()
        and move_count < MAX_MOVES
    ):

        is_model_turn = (
            board.turn == chess.WHITE
            and model_white
        ) or (
            board.turn == chess.BLACK
            and not model_white
        )

        if is_model_turn:

            if model_type == "v1":
                move = v1_move(
                    model,
                    board
                )

            elif model_type == "v2":
                move = v2_move(
                    model,
                    board
                )

            else:
                move = masked_move(
                    model,
                    board
                )

        else:

            move = stockfish_move(
                board,
                engine
            )

        if move is None:
            break

        board.push(move)

        move_count += 1

    return (
        board.result(
            claim_draw=True
        ),
        move_count
    )


def outcome_for_model(
    result,
    model_white
):

    if result == "1/2-1/2":
        return "draw"

    model_won = (
        result == "1-0"
        if model_white
        else result == "0-1"
    )

    return "win" if model_won else "loss"


def elo_difference(
    wins,
    draws,
    losses
):

    total = (
        wins
        + draws
        + losses
    )

    score = (
        wins
        + 0.5 * draws
    ) / total

    score = max(
        0.01,
        min(0.99, score)
    )

    return 400 * math.log10(
        score / (1 - score)
    )


def benchmark(
    name,
    model,
    model_type,
    engine
):

    wins = 0
    draws = 0
    losses = 0

    lengths = []

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for game in range(1, GAMES + 1):

        model_white = (
            game % 2 == 1
        )

        result, moves = play_game(
            model,
            model_type,
            engine,
            model_white
        )

        outcome = outcome_for_model(
            result,
            model_white
        )

        if outcome == "win":
            wins += 1

        elif outcome == "draw":
            draws += 1

        else:
            losses += 1

        lengths.append(moves)

        print(
            f"Game {game:02d} | "
            f"{'White' if model_white else 'Black':5s} | "
            f"{result:7s} | "
            f"{outcome.upper():5s} | "
            f"moves={moves}"
        )

    total = (
        wins
        + draws
        + losses
    )

    score = (
        wins
        + 0.5 * draws
    ) / total

    elo = elo_difference(
        wins,
        draws,
        losses
    )

    print("\nResults:")
    print("Wins:", wins)
    print("Draws:", draws)
    print("Losses:", losses)
    print(f"Score: {score:.3f}")
    print(f"Win rate: {wins / total:.1%}")
    print(f"Draw rate: {draws / total:.1%}")
    print(f"Loss rate: {losses / total:.1%}")
    print(
        f"Average game length: "
        f"{np.mean(lengths):.1f}"
    )
    print(
        f"Approx. Elo difference: "
        f"{elo:+.0f}"
    )

    return {
        "name": name,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "elo": elo,
        "avg_moves": float(
            np.mean(lengths)
        )
    }


def main():

    print("=" * 70)
    print("AI CHESS AGENT - V2.9 MODEL BENCHMARK")
    print("=" * 70)

    stockfish_path = find_stockfish()

    print(
        "\nStockfish:",
        stockfish_path
    )

    print(
        "Stockfish depth:",
        STOCKFISH_DEPTH
    )

    print(
        "Games per model:",
        GAMES
    )

    engine = (
        chess.engine.SimpleEngine
        .popen_uci(stockfish_path)
    )

    engine.configure({
        "Skill Level": 5
    })

    print("\nLoading V1...")
    v1 = PPO.load(
        V1_MODEL
    )

    print("Loading V2...")
    v2 = PPO.load(
        V2_MODEL
    )

    print("Loading masked V2...")
    masked = MaskablePPO.load(
        MASKED_MODEL
    )

    results = []

    results.append(
        benchmark(
            "V1 PPO",
            v1,
            "v1",
            engine
        )
    )

    results.append(
        benchmark(
            "V2 PPO",
            v2,
            "v2",
            engine
        )
    )

    results.append(
        benchmark(
            "V2.8 Maskable PPO",
            masked,
            "masked",
            engine
        )
    )

    engine.quit()

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':25s}"
        f"{'W':>5s}"
        f"{'D':>5s}"
        f"{'L':>5s}"
        f"{'Score':>9s}"
        f"{'Elo':>9s}"
    )

    print("-" * 70)

    for r in results:

        print(
            f"{r['name']:25s}"
            f"{r['wins']:5d}"
            f"{r['draws']:5d}"
            f"{r['losses']:5d}"
            f"{r['score']:9.3f}"
            f"{r['elo']:9.0f}"
        )

    print("=" * 70)

    best = max(
        results,
        key=lambda x: x["score"]
    )

    print(
        "\nBest model:",
        best["name"]
    )

    print(
        f"Best score: "
        f"{best['score']:.3f}"
    )


if __name__ == "__main__":
    main()
