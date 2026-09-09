import os
import chess
import chess.engine


STOCKFISH_PATHS = [
    "/opt/homebrew/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/usr/games/stockfish",
]


PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


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


def material_score(board):
    """
    Positive = White has more material.
    Negative = Black has more material.
    """

    score = 0.0

    for piece in board.piece_map().values():

        value = PIECE_VALUES[
            piece.piece_type
        ]

        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value

    return score


def material_breakdown(board):

    white = {
        "pawn": 0,
        "knight": 0,
        "bishop": 0,
        "rook": 0,
        "queen": 0,
    }

    black = {
        "pawn": 0,
        "knight": 0,
        "bishop": 0,
        "rook": 0,
        "queen": 0,
    }

    names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
    }

    for piece in board.piece_map().values():

        if piece.piece_type == chess.KING:
            continue

        name = names[piece.piece_type]

        if piece.color == chess.WHITE:
            white[name] += 1
        else:
            black[name] += 1

    return {
        "white": white,
        "black": black,
    }


def captured_piece(board_before, move):

    if not board_before.is_capture(move):
        return None

    if board_before.is_en_passant(move):
        return "pawn"

    piece = board_before.piece_at(
        move.to_square
    )

    if piece is None:
        return None

    names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }

    return names[piece.piece_type]


def analyze_move(
    board,
    move,
    engine,
    depth=12,
):
    """
    Analyze a candidate move using Stockfish.

    Returns:
        best move
        played move evaluation
        best move evaluation
        centipawn loss
        explanation
    """

    if move not in board.legal_moves:
        raise ValueError(
            "Move is not legal in this position."
        )

    board_before = board.copy()

    # ---------------------------------------------------------
    # Best move
    # ---------------------------------------------------------

    best_result = engine.analyse(
        board,
        chess.engine.Limit(
            depth=depth
        ),
        multipv=1,
    )

    # python-chess may return either:
    #   dict       -> single PV
    #   list[dict] -> MultiPV result
    if isinstance(best_result, list):
        best_result = best_result[0]

    best_move = best_result["pv"][0]

    best_score = best_result["score"].pov(
        board.turn
    )

    best_cp = best_score.score(
        mate_score=100000
    )

    # ---------------------------------------------------------
    # Played move
    # ---------------------------------------------------------

    played_board = board.copy()

    san = played_board.san(move)

    is_capture = played_board.is_capture(
        move
    )

    captured = captured_piece(
        played_board,
        move
    )

    played_board.push(move)

    played_result = engine.analyse(
        played_board,
        chess.engine.Limit(
            depth=depth
        )
    )

    played_score = (
        played_result["score"]
        .pov(board.turn)
    )

    played_cp = played_score.score(
        mate_score=100000
    )

    # ---------------------------------------------------------
    # Centipawn loss
    # ---------------------------------------------------------

    cp_loss = max(
        0,
        best_cp - played_cp
    )

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    if played_board.is_checkmate():

        classification = "checkmate"

    elif cp_loss <= 20:

        classification = "excellent"

    elif cp_loss <= 50:

        classification = "good"

    elif cp_loss <= 100:

        classification = "inaccuracy"

    elif cp_loss <= 250:

        classification = "mistake"

    else:

        classification = "blunder"

    # ---------------------------------------------------------
    # Explanation
    # ---------------------------------------------------------

    reasons = []

    if played_board.is_checkmate():
        reasons.append(
            "The move delivers checkmate."
        )

    elif played_board.is_check():
        reasons.append(
            "The move gives check."
        )

    if is_capture and captured:
        reasons.append(
            f"The move captures a {captured}."
        )

    if move == best_move:
        reasons.append(
            "Stockfish considers this the best move."
        )

    elif cp_loss <= 20:
        reasons.append(
            "The move is very close to the engine's "
            "best continuation."
        )

    elif cp_loss <= 100:
        reasons.append(
            "The move gives up a small amount of "
            "evaluation."
        )

    elif cp_loss <= 250:
        reasons.append(
            "The move loses noticeable evaluation."
        )

    else:
        reasons.append(
            "The move loses significant evaluation."
        )

    explanation = " ".join(
        reasons
    )

    return {
        "move": move.uci(),
        "san": san,
        "best_move": best_move.uci(),
        "best_move_san": board.san(
            best_move
        ),
        "classification": classification,
        "centipawn_loss": cp_loss,
        "best_evaluation_cp": best_cp,
        "played_evaluation_cp": played_cp,
        "is_capture": is_capture,
        "captured_piece": captured,
        "gives_check": played_board.is_check(),
        "is_checkmate": played_board.is_checkmate(),
        "material_before": material_score(
            board_before
        ),
        "material_after": material_score(
            played_board
        ),
        "material": material_breakdown(
            played_board
        ),
        "explanation": explanation,
    }


def analyze_position(
    board,
    engine,
    depth=12,
):
    """
    Analyze the current board position.
    """

    result = engine.analyse(
        board,
        chess.engine.Limit(
            depth=depth
        )
    )

    best_move = result["pv"][0]

    score = result["score"].pov(
        chess.WHITE
    )

    cp = score.score(
        mate_score=100000
    )

    return {
        "fen": board.fen(),
        "turn": (
            "white"
            if board.turn == chess.WHITE
            else "black"
        ),
        "best_move": best_move.uci(),
        "best_move_san": board.san(
            best_move
        ),
        "evaluation_cp": cp,
        "evaluation_pawns": cp / 100.0,
        "material_score": material_score(
            board
        ),
        "material": material_breakdown(
            board
        ),
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),
    }


if __name__ == "__main__":

    print("=" * 70)
    print("CHESS AI EXPLAINABILITY TEST")
    print("=" * 70)

    path = find_stockfish()

    print("\nStockfish:", path)

    engine = chess.engine.SimpleEngine.popen_uci(
        path
    )

    board = chess.Board()

    print("\nStarting position:")

    print(board)

    print("\nAnalyzing position...")

    position = analyze_position(
        board,
        engine,
        depth=10
    )

    print(
        "\nBest move:",
        position["best_move_san"]
    )

    print(
        "Evaluation:",
        position["evaluation_pawns"]
    )

    move = chess.Move.from_uci(
        "e2e4"
    )

    print("\nAnalyzing e4...")

    analysis = analyze_move(
        board,
        move,
        engine,
        depth=10
    )

    print(
        "\nAI move:",
        analysis["san"]
    )

    print(
        "Engine best:",
        analysis["best_move_san"]
    )

    print(
        "Classification:",
        analysis["classification"]
    )

    print(
        "Centipawn loss:",
        analysis["centipawn_loss"]
    )

    print(
        "Explanation:",
        analysis["explanation"]
    )

    engine.quit()

    print("\n" + "=" * 70)
    print("EXPLAINABILITY TEST COMPLETE")
    print("=" * 70)
