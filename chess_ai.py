"""
AI Chess Agent V2
Game engine + PPO baseline + Stockfish integration.
"""

from __future__ import annotations

import os
import shutil
import threading
from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.engine
import numpy as np
from stable_baselines3 import PPO


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chess_ai_agent.h5",
)


def find_stockfish() -> str:
    """Find Stockfish on common macOS/Linux installation paths."""
    candidates = [
        os.environ.get("STOCKFISH_PATH"),
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/games/stockfish",
        shutil.which("stockfish"),
    ]

    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "Stockfish binary not found. Set STOCKFISH_PATH or install Stockfish."
    )


def board_to_legacy_state(
    board: chess.Board,
    perspective: chess.Color = chess.WHITE,
) -> np.ndarray:
    """
    Preserve the original PPO model's 8x8x12 observation format.

    When the PPO agent plays Black, mirror the board so the old
    White-trained model receives a White-perspective representation.
    """
    working_board = board.copy(stack=False)

    if perspective == chess.BLACK:
        working_board = working_board.mirror()

    state = np.zeros((8, 8, 12), dtype=np.uint8)

    for square, piece in working_board.piece_map().items():
        row, col = divmod(square, 8)
        piece_idx = (piece.piece_type - 1) + (
            6 if piece.color == chess.BLACK else 0
        )
        state[row, col, piece_idx] = 1

    return state


def mirror_move(move: chess.Move) -> chess.Move:
    """Convert a move between normal and mirrored board coordinates."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


class PPOAgent:
    """Wrapper around the existing trained PPO model."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = PPO.load(model_path)

    def choose_move(self, board: chess.Board) -> chess.Move:
        """
        Use the existing PPO model.

        The legacy model has 4672 actions and the original project maps
        actions to legal moves. We retain that compatibility here.
        """
        if board.is_game_over():
            raise ValueError("Cannot choose a move from a finished game.")

        perspective = board.turn
        state = board_to_legacy_state(board, perspective)

        action, _ = self.model.predict(state, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])

        if perspective == chess.WHITE:
            legal_moves = list(board.legal_moves)
            return legal_moves[action % len(legal_moves)]

        # Black: predict on mirrored board and transform selected move back.
        mirrored = board.mirror()
        mirrored_moves = list(mirrored.legal_moves)
        selected = mirrored_moves[action % len(mirrored_moves)]
        return mirror_move(selected)


class StockfishAgent:
    """Stockfish wrapper with configurable skill level."""

    def __init__(
        self,
        skill_level: int = 8,
        move_time: float = 0.25,
    ):
        self.skill_level = max(0, min(20, int(skill_level)))
        self.move_time = max(0.05, float(move_time))
        self.path = find_stockfish()
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path)

        try:
            self.engine.configure({"Skill Level": self.skill_level})
        except Exception:
            pass

    def choose_move(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(
            board,
            chess.engine.Limit(time=self.move_time),
        )
        return result.move

    def evaluate(self, board: chess.Board) -> Optional[float]:
        """Return evaluation in approximate pawn units from White's view."""
        if board.is_game_over():
            return None

        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(time=0.15),
            )
            score = info["score"].pov(chess.WHITE).score(mate_score=100000)
            return round(score / 100.0, 2)
        except Exception:
            return None

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


@dataclass
class Game:
    mode: str = "human_vs_ppo"
    human_color: chess.Color = chess.BLACK
    stockfish_level: int = 8
    board: chess.Board = field(default_factory=chess.Board)
    move_history: list[str] = field(default_factory=list)
    san_history: list[str] = field(default_factory=list)
    captured: list[str] = field(default_factory=list)
    snapshots: list[chess.Board] = field(default_factory=list)

    def reset(self):
        self.board = chess.Board()
        self.move_history.clear()
        self.san_history.clear()
        self.captured.clear()
        self.snapshots.clear()

    @property
    def ai_color(self) -> chess.Color:
        return not self.human_color

    def is_human_turn(self) -> bool:
        if self.mode == "human_vs_ppo" or self.mode == "human_vs_stockfish":
            return self.board.turn == self.human_color
        return False

    def push_move(self, move: chess.Move) -> str:
        if move not in self.board.legal_moves:
            raise ValueError("Illegal chess move.")

        san = self.board.san(move)

        if self.board.is_capture(move):
            captured_piece = self.board.piece_at(move.to_square)

            # En passant capture.
            if captured_piece is None and self.board.is_en_passant(move):
                captured_piece = chess.Piece(
                    chess.PAWN,
                    not self.board.turn,
                )

            if captured_piece:
                self.captured.append(captured_piece.symbol())

        self.snapshots.append(self.board.copy(stack=True))
        self.board.push(move)

        self.move_history.append(move.uci())
        self.san_history.append(san)

        return san

    def undo(self) -> bool:
        if not self.snapshots:
            return False

        self.board = self.snapshots.pop()

        if self.move_history:
            self.move_history.pop()

        if self.san_history:
            self.san_history.pop()

        # Recalculate captures safely.
        self.captured.clear()
        replay = chess.Board()

        for uci in self.move_history:
            move = chess.Move.from_uci(uci)

            if replay.is_capture(move):
                piece = replay.piece_at(move.to_square)

                if piece is None and replay.is_en_passant(move):
                    piece = chess.Piece(chess.PAWN, not replay.turn)

                if piece:
                    self.captured.append(piece.symbol())

            replay.push(move)

        return True


class ChessGameManager:
    """Thread-safe local game manager."""

    def __init__(self):
        self.lock = threading.RLock()
        self.ppo = PPOAgent()
        self.stockfish: Optional[StockfishAgent] = None
        self.game = Game()

    def new_game(
        self,
        mode: str = "human_vs_ppo",
        human_color: str = "black",
        stockfish_level: int = 8,
    ):
        with self.lock:
            allowed_modes = {
                "human_vs_ppo",
                "human_vs_stockfish",
                "ppo_vs_stockfish",
            }

            if mode not in allowed_modes:
                raise ValueError("Invalid game mode.")

            color = (
                chess.WHITE
                if human_color.lower() == "white"
                else chess.BLACK
            )

            self.game = Game(
                mode=mode,
                human_color=color,
                stockfish_level=int(stockfish_level),
            )

            self._reset_stockfish()

            if mode in {"human_vs_stockfish", "ppo_vs_stockfish"}:
                self.stockfish = StockfishAgent(
                    skill_level=stockfish_level
                )

            # If AI is White, make its first move.
            if mode == "ppo_vs_stockfish":
                self._ai_turns()

            elif mode == "human_vs_ppo" and color == chess.BLACK:
                self._ppo_move()

            elif mode == "human_vs_stockfish" and color == chess.BLACK:
                self._stockfish_move()

            return self.state()

    def _reset_stockfish(self):
        if self.stockfish:
            self.stockfish.close()
            self.stockfish = None

    def _ppo_move(self):
        move = self.ppo.choose_move(self.game.board)
        self.game.push_move(move)

    def _stockfish_move(self):
        if not self.stockfish:
            self.stockfish = StockfishAgent(
                skill_level=self.game.stockfish_level
            )

        move = self.stockfish.choose_move(self.game.board)
        self.game.push_move(move)

    def _ai_turns(self):
        """Play AI-vs-AI until the game ends or a safety move limit is hit."""
        safety_limit = 120

        while not self.game.board.is_game_over() and len(
            self.game.move_history
        ) < safety_limit:
            if self.game.board.turn == chess.WHITE:
                self._ppo_move()
            else:
                self._stockfish_move()

    def make_human_move(self, uci: str):
        with self.lock:
            if not self.game.is_human_turn():
                raise ValueError("It is not the human player's turn.")

            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                raise ValueError("Invalid move format.")

            san = self.game.push_move(move)

            # AI response.
            if not self.game.board.is_game_over():
                if self.game.mode == "human_vs_ppo":
                    if self.game.board.turn == self.game.ai_color:
                        self._ppo_move()

                elif self.game.mode == "human_vs_stockfish":
                    if self.game.board.turn == self.game.ai_color:
                        self._stockfish_move()

            return san

    def ai_move(self):
        with self.lock:
            if self.game.board.is_game_over():
                return self.state()

            if self.game.mode == "human_vs_ppo":
                if self.game.board.turn == self.game.ai_color:
                    self._ppo_move()

            elif self.game.mode == "human_vs_stockfish":
                if self.game.board.turn == self.game.ai_color:
                    self._stockfish_move()

            elif self.game.mode == "ppo_vs_stockfish":
                self._ai_turns()

            return self.state()

    def undo(self):
        with self.lock:
            if self.game.mode.startswith("human"):
                # Undo the human move plus AI response where possible.
                self.game.undo()

                if self.game.move_history:
                    self.game.undo()

            else:
                self.game.undo()

            return self.state()

    def state(self):
        with self.lock:
            board = self.game.board

            status = "Playing"

            if board.is_checkmate():
                winner = "White" if board.turn == chess.BLACK else "Black"
                status = f"Checkmate — {winner} wins"
            elif board.is_stalemate():
                status = "Draw — Stalemate"
            elif board.is_insufficient_material():
                status = "Draw — Insufficient material"
            elif board.is_seventyfive_moves():
                status = "Draw — 75-move rule"
            elif board.is_fivefold_repetition():
                status = "Draw — Fivefold repetition"
            elif board.is_check():
                status = f"Check — {'White' if board.turn else 'Black'} to move"

            legal_moves = [move.uci() for move in board.legal_moves]

            evaluation = None

            if self.stockfish and not board.is_game_over():
                evaluation = self.stockfish.evaluate(board)

            return {
                "fen": board.fen(),
                "turn": "white" if board.turn == chess.WHITE else "black",
                "status": status,
                "game_over": board.is_game_over(),
                "legal_moves": legal_moves,
                "moves": self.game.san_history,
                "uci_moves": self.game.move_history,
                "captured": self.game.captured,
                "fullmove": board.fullmove_number,
                "halfmove": board.halfmove_clock,
                "mode": self.game.mode,
                "human_color": (
                    "white"
                    if self.game.human_color == chess.WHITE
                    else "black"
                ),
                "ai_color": (
                    "white"
                    if self.game.ai_color == chess.WHITE
                    else "black"
                ),
                "evaluation": evaluation,
            }

    def close(self):
        self._reset_stockfish()


manager = ChessGameManager()


def play_game():
    """Compatibility function for the original application."""
    result = manager.new_game(
        mode="ppo_vs_stockfish",
        stockfish_level=8,
    )
    return result["uci_moves"]
