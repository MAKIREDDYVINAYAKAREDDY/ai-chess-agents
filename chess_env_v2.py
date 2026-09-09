import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ChessEnvV2(gym.Env):
    """
    Improved Chess RL environment.

    Observation:
        8x8x18

        Channels 0-11:
            White pieces:
                0 Pawn
                1 Knight
                2 Bishop
                3 Rook
                4 Queen
                5 King

            Black pieces:
                6 Pawn
                7 Knight
                8 Bishop
                9 Rook
                10 Queen
                11 King

        Channel 12:
            Side to move

        Channels 13-16:
            Castling rights
                13 White kingside
                14 White queenside
                15 Black kingside
                16 Black queenside

        Channel 17:
            En-passant square

    The old V1 model is NOT used or modified here.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_moves=200, render_mode=None):
        super().__init__()

        self.max_moves = max_moves
        self.render_mode = render_mode

        self.board = chess.Board()
        self.move_count = 0

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8, 8, 18),
            dtype=np.float32,
        )

        # Keep the V1 action-space size for now.
        # V2.3 will replace this with proper move encoding.
        self.action_space = spaces.Discrete(4672)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.board = chess.Board()
        self.move_count = 0

        observation = self._get_observation()

        info = {
            "fen": self.board.fen(),
            "turn": "white",
        }

        return observation, info

    def step(self, action):
        legal_moves = list(self.board.legal_moves)

        if not legal_moves:
            return (
                self._get_observation(),
                0.0,
                True,
                False,
                {
                    "fen": self.board.fen(),
                    "result": self.board.result(),
                },
            )

        # Temporary V2 compatibility mapping.
        action = int(action)
        move = legal_moves[action % len(legal_moves)]

        san = self.board.san(move)
        self.board.push(move)

        self.move_count += 1

        terminated = False
        truncated = False
        reward = 0.0

        # Terminal game result
        if self.board.is_checkmate():
            terminated = True

            # The player who just moved delivered checkmate.
            reward = 1.0

        elif self.board.is_stalemate():
            terminated = True
            reward = 0.5

        elif self.board.is_insufficient_material():
            terminated = True
            reward = 0.5

        elif self.board.is_fivefold_repetition():
            terminated = True
            reward = 0.5

        elif self.board.is_seventyfive_moves():
            terminated = True
            reward = 0.5

        elif self.board.is_variant_loss():
            terminated = True
            reward = -1.0

        elif self.board.is_variant_win():
            terminated = True
            reward = 1.0

        # Prevent indefinitely long episodes.
        if self.move_count >= self.max_moves and not terminated:
            truncated = True

        observation = self._get_observation()

        info = {
            "move": move.uci(),
            "san": san,
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "legal_moves": len(list(self.board.legal_moves)),
            "result": self.board.result() if terminated else "*",
        }

        if self.render_mode == "human":
            print(self.board)

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        state = np.zeros((8, 8, 18), dtype=np.float32)

        # ---------------------------------------------------------
        # Piece planes
        # ---------------------------------------------------------
        for square, piece in self.board.piece_map().items():
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)

            piece_index = piece.piece_type - 1

            if piece.color == chess.BLACK:
                piece_index += 6

            state[row, col, piece_index] = 1.0

        # ---------------------------------------------------------
        # Side to move
        # ---------------------------------------------------------
        if self.board.turn == chess.WHITE:
            state[:, :, 12] = 1.0

        # ---------------------------------------------------------
        # Castling rights
        # ---------------------------------------------------------
        if self.board.has_kingside_castling_rights(chess.WHITE):
            state[:, :, 13] = 1.0

        if self.board.has_queenside_castling_rights(chess.WHITE):
            state[:, :, 14] = 1.0

        if self.board.has_kingside_castling_rights(chess.BLACK):
            state[:, :, 15] = 1.0

        if self.board.has_queenside_castling_rights(chess.BLACK):
            state[:, :, 16] = 1.0

        # ---------------------------------------------------------
        # En-passant square
        # ---------------------------------------------------------
        if self.board.ep_square is not None:
            square = self.board.ep_square

            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)

            state[row, col, 17] = 1.0

        return state

    def legal_moves(self):
        """Return all currently legal moves."""
        return list(self.board.legal_moves)

    def legal_moves_uci(self):
        """Return legal moves in UCI notation."""
        return [move.uci() for move in self.board.legal_moves]

    def get_fen(self):
        return self.board.fen()

    def render(self):
        print(self.board)
        print()
        print(f"FEN: {self.board.fen()}")
        print(
            "Turn:",
            "White" if self.board.turn == chess.WHITE else "Black"
        )

    def close(self):
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("ChessEnvV2 Test")
    print("=" * 60)

    env = ChessEnvV2()

    observation, info = env.reset()

    print("\nObservation shape:")
    print(observation.shape)

    print("\nObservation dtype:")
    print(observation.dtype)

    print("\nAction space:")
    print(env.action_space)

    print("\nObservation space:")
    print(env.observation_space)

    print("\nInitial FEN:")
    print(info["fen"])

    print("\nInitial legal moves:")
    print(len(env.legal_moves()))

    print("\nFirst 10 legal moves:")
    print(env.legal_moves_uci()[:10])

    print("\nPlaying 5 random moves...")

    for i in range(5):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        print(
            f"Move {i + 1}: "
            f"{info['san']} "
            f"({info['move']}) | "
            f"Reward: {reward}"
        )

        if terminated or truncated:
            break

    print("\nFinal FEN:")
    print(env.get_fen())

    print("\nEnvironment test completed successfully.")

    env.close()
