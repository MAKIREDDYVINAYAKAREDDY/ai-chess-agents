from flask import Flask, jsonify, render_template, request
import chess
import chess.engine
import os
import shutil

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

from chess_env_masked import MaskedChessEnv
from chess_analysis import analyze_position, analyze_move


app = Flask(__name__)


# ============================================================
# STOCKFISH
# ============================================================

def find_stockfish():
    candidates = [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/games/stockfish",
        shutil.which("stockfish"),
    ]

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    return None


STOCKFISH_PATH = find_stockfish()
engine = None

if STOCKFISH_PATH:
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print(f"Stockfish found: {STOCKFISH_PATH}")
    except Exception as e:
        print(f"Stockfish startup error: {e}")
else:
    print("WARNING: Stockfish not found.")


# ============================================================
# PPO
# ============================================================

PPO_MODEL_PATH = "chess_ai_agent_v2_masked.zip"

ppo_model = None

if os.path.exists(PPO_MODEL_PATH):
    try:
        ppo_model = MaskablePPO.load(PPO_MODEL_PATH)
        print(f"PPO model loaded: {PPO_MODEL_PATH}")
    except Exception as e:
        print(f"PPO model loading error: {e}")
else:
    print(f"WARNING: PPO model not found: {PPO_MODEL_PATH}")


# ============================================================
# GAME STATE
# ============================================================

board = chess.Board()
move_history = []

game_mode = "human_vs_stockfish"
human_color = "white"
stockfish_level = 5


# ============================================================
# STATE
# ============================================================

def build_state():
    legal_moves = [move.uci() for move in board.legal_moves]

    return {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",

        "mode": game_mode,
        "human_color": human_color,
        "stockfish_level": stockfish_level,

        "legal_moves": legal_moves,

        "game_over": board.is_game_over(),
        "is_check": board.is_check(),
        "is_checkmate": board.is_checkmate(),
        "is_stalemate": board.is_stalemate(),

        "moves": move_history,

        "last_move": (
            move_history[-1]
            if move_history
            else None
        ),

        "fullmove_number": board.fullmove_number,
        "halfmove_clock": board.halfmove_clock,
    }


def response_state():
    return jsonify({
        "success": True,
        "state": build_state()
    })


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# NEW GAME
# ============================================================

@app.route("/api/new", methods=["POST"])
def new_game():
    global board
    global move_history
    global game_mode
    global human_color
    global stockfish_level

    data = request.get_json(silent=True) or {}

    game_mode = data.get(
        "mode",
        "human_vs_stockfish"
    )

    human_color = data.get(
        "human_color",
        "white"
    )

    stockfish_level = int(
        data.get(
            "stockfish_level",
            5
        )
    )

    board = chess.Board()
    move_history = []

    return response_state()


# ============================================================
# STATE
# ============================================================

@app.route("/api/state", methods=["GET"])
def get_state():
    return response_state()


# ============================================================
# HUMAN MOVE
# ============================================================

@app.route("/api/move", methods=["POST"])
def make_move():
    global board
    global move_history

    data = request.get_json(silent=True) or {}

    uci = data.get("uci")

    if not uci:
        uci = data.get("move")

    if not uci:
        return jsonify({
            "success": False,
            "error": "No move supplied"
        }), 400

    try:
        move = chess.Move.from_uci(uci)

    except ValueError:
        return jsonify({
            "success": False,
            "error": f"Invalid move: {uci}"
        }), 400

    if move not in board.legal_moves:
        return jsonify({
            "success": False,
            "error": f"Illegal move: {uci}"
        }), 400

    san = board.san(move)

    board.push(move)

    move_history.append({
        "uci": uci,
        "san": san,
        "player": "human"
    })

    return jsonify({
        "success": True,
        "state": build_state()
    })


# ============================================================
# UNDO
# ============================================================

@app.route("/api/undo", methods=["POST"])
def undo():
    global board
    global move_history

    if not board.move_stack:
        return jsonify({
            "success": False,
            "error": "No moves to undo"
        }), 400

    board.pop()

    if move_history:
        move_history.pop()

    return response_state()


# ============================================================
# PPO MOVE
# ============================================================

def get_ppo_move():
    if ppo_model is None:
        raise RuntimeError(
            "PPO model is not loaded"
        )

    env = MaskedChessEnv(
        max_moves=200
    )

    try:
        env.board = board.copy()
        env.move_count = len(board.move_stack)

        observation = env._get_observation()

        action_masks = env.action_masks()

        action, _ = ppo_model.predict(
            observation,
            deterministic=True,
            action_masks=action_masks
        )

        action = int(action)

        move = env.action_to_move(action)

        if (
            move is None
            or move not in board.legal_moves
        ):
            legal_moves = list(board.legal_moves)

            if not legal_moves:
                return None

            print(
                "WARNING: PPO produced invalid action:",
                action
            )

            move = legal_moves[0]

        return move

    finally:
        env.close()


@app.route("/api/ppo-move", methods=["POST"])
def ppo_move():
    global board
    global move_history

    if board.is_game_over():
        return jsonify({
            "success": False,
            "error": "Game is already over"
        }), 400

    try:
        move = get_ppo_move()

        if move is None:
            return jsonify({
                "success": False,
                "error": "PPO returned no legal move"
            }), 500

        uci = move.uci()
        san = board.san(move)

        board.push(move)

        move_history.append({
            "uci": uci,
            "san": san,
            "player": "ppo"
        })

        return jsonify({
            "success": True,
            "state": build_state()
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# STOCKFISH MOVE
# ============================================================

@app.route("/api/ai-move", methods=["POST"])
def ai_move():
    global board
    global move_history

    if board.is_game_over():
        return jsonify({
            "success": False,
            "error": "Game is already over"
        }), 400

    if engine is None:
        return jsonify({
            "success": False,
            "error": "Stockfish is not available"
        }), 503

    try:
        data = request.get_json(
            silent=True
        ) or {}

        level = int(
            data.get(
                "stockfish_level",
                stockfish_level
            )
        )

        level = max(
            0,
            min(20, level)
        )

        engine.configure({
            "Skill Level": level
        })

        result = engine.play(
            board,
            chess.engine.Limit(
                depth=5
            )
        )

        move = result.move

        if move is None:
            return jsonify({
                "success": False,
                "error": "Stockfish returned no move"
            }), 500

        uci = move.uci()
        san = board.san(move)

        board.push(move)

        move_history.append({
            "uci": uci,
            "san": san,
            "player": "stockfish"
        })

        return jsonify({
            "success": True,
            "state": build_state()
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# AI vs AI
# ============================================================

@app.route("/api/ai-vs-ai", methods=["POST"])
def ai_vs_ai():
    global board
    global move_history

    if game_mode != "ppo_vs_stockfish":
        return jsonify({
            "success": False,
            "error": "AI vs AI endpoint requires PPO vs Stockfish mode"
        }), 400

    if board.is_game_over():
        return response_state()

    try:

        if board.turn == chess.WHITE:

            move = get_ppo_move()

            if move is None:
                raise RuntimeError(
                    "PPO returned no legal move"
                )

            player = "ppo"

        else:

            if engine is None:
                raise RuntimeError(
                    "Stockfish is not available"
                )

            engine.configure({
                "Skill Level": stockfish_level
            })

            result = engine.play(
                board,
                chess.engine.Limit(
                    depth=5
                )
            )

            move = result.move
            player = "stockfish"

        uci = move.uci()
        san = board.san(move)

        board.push(move)

        move_history.append({
            "uci": uci,
            "san": san,
            "player": player
        })

        return jsonify({
            "success": True,
            "state": build_state()
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# ANALYSIS
# ============================================================

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(
        silent=True
    ) or {}

    fen = data.get(
        "fen",
        board.fen()
    )

    try:
        result = analyze_position(fen)

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/analyze-move", methods=["POST"])
def api_analyze_move():
    data = request.get_json(
        silent=True
    ) or {}

    fen = data.get(
        "fen",
        board.fen()
    )

    move = data.get("move")

    if not move:
        return jsonify({
            "success": False,
            "error": "No move supplied"
        }), 400

    try:
        result = analyze_move(
            fen,
            move
        )

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# STATUS
# ============================================================

@app.route("/api/engine-status", methods=["GET"])
def engine_status():
    return jsonify({
        "stockfish": engine is not None,
        "ppo": ppo_model is not None,
        "stockfish_path": STOCKFISH_PATH,
        "ppo_model": PPO_MODEL_PATH
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "stockfish": engine is not None,
        "ppo": ppo_model is not None
    })


# ============================================================
# OLD COMPATIBILITY
# ============================================================

@app.route("/start_game", methods=["GET"])
def start_game():
    global board
    global move_history

    board = chess.Board()
    move_history = []

    moves = []

    if engine:

        for _ in range(20):

            if board.is_game_over():
                break

            result = engine.play(
                board,
                chess.engine.Limit(
                    depth=3
                )
            )

            move = result.move

            moves.append(
                move.uci()
            )

            board.push(move)

    return jsonify({
        "moves": moves
    })


# ============================================================
# SHUTDOWN
# ============================================================

@app.route("/api/quit-engine", methods=["POST"])
def quit_engine():
    global engine

    if engine:

        try:
            engine.quit()
        except Exception:
            pass

        engine = None

    return jsonify({
        "success": True
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:
        app.run(debug=True)

    finally:

        if engine:

            try:
                engine.quit()
            except Exception:
                pass
