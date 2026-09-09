import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ChessEnvV3(gym.Env):
    """
    V2.3 Chess RL environment.

    Improvements over V2.2:
    - Fixed action encoding
    - Deterministic move -> action mapping
    - Deterministic action -> move mapping
    - Legal-action masking
    - Supports promotions
    - Supports castling
    - Supports en-passant

    Action encoding:
        action = from_square * 64 + to_square

    Promotion moves use additional promotion slots.
    """

    metadata = {"render_modes": ["human"]}

    # 64 * 64 = 4096 normal moves
    # Additional promotion actions:
    # Queen, Rook, Bishop, Knight
    # 64 * 64 * 4 = 16384 possible promotion combinations,
    # but we only need a compact fixed action space.

    NORMAL_ACTIONS = 4096
    PROMOTION_ACTIONS = 4096 * 4
    ACTION_SIZE = NORMAL_ACTIONS + PROMOTION_ACTIONS

    PROMOTION_OFFSET = NORMAL_ACTIONS

    PROMOTION_PIECES = {
        chess.QUEEN: 0,
        chess.ROOK: 1,
        chess.BISHOP: 2,
        chess.KNIGHT: 3,
    }

    REVERSE_PROMOTION = {
        0: chess.QUEEN,
        1: chess.ROOK,
        2: chess.BISHOP,
        3: chess.KNIGHT,
    }

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

        self.action_space = spaces.Discrete(self.ACTION_SIZE)

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.board = chess.Board()
        self.move_count = 0

        return self._get_observation(), {
            "fen": self.board.fen(),
            "turn": "white",
        }

    def step(self, action):
        action = int(action)

        move = self.action_to_move(action)

        # Invalid action
        if move is None or move not in self.board.legal_moves:
            return (
                self._get_observation(),
                -0.1,
                False,
                False,
                {
                    "invalid_action": True,
                    "action": action,
                    "fen": self.board.fen(),
                },
            )

        san = self.board.san(move)

        self.board.push(move)
        self.move_count += 1

        terminated = False
        truncated = False
        reward = 0.0

        if self.board.is_checkmate():
            terminated = True
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

        if self.move_count >= self.max_moves and not terminated:
            truncated = True

        info = {
            "action": action,
            "move": move.uci(),
            "san": san,
            "fen": self.board.fen(),
            "turn": (
                "white"
                if self.board.turn == chess.WHITE
                else "black"
            ),
            "legal_moves": len(list(self.board.legal_moves)),
            "result": self.board.result() if terminated else "*",
            "invalid_action": False,
        }

        if self.render_mode == "human":
            self.render()

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    # ---------------------------------------------------------
    # Observation
    # ---------------------------------------------------------

    def _get_observation(self):
        state = np.zeros(
            (8, 8, 18),
            dtype=np.float32,
        )

        for square, piece in self.board.piece_map().items():

            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)

            piece_index = piece.piece_type - 1

            if piece.color == chess.BLACK:
                piece_index += 6

            state[row, col, piece_index] = 1.0

        # Side to move
        if self.board.turn == chess.WHITE:
            state[:, :, 12] = 1.0

        # Castling rights
        if self.board.has_kingside_castling_rights(
            chess.WHITE
        ):
            state[:, :, 13] = 1.0

        if self.board.has_queenside_castling_rights(
            chess.WHITE
        ):
            state[:, :, 14] = 1.0

        if self.board.has_kingside_castling_rights(
            chess.BLACK
        ):
            state[:, :, 15] = 1.0

        if self.board.has_queenside_castling_rights(
            chess.BLACK
        ):
            state[:, :, 16] = 1.0

        # En-passant square
        if self.board.ep_square is not None:

            square = self.board.ep_square

            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)

            state[row, col, 17] = 1.0

        return state

    # ---------------------------------------------------------
    # Action Encoding
    # ---------------------------------------------------------

    @classmethod
    def move_to_action(cls, move):
        """
        Convert chess.Move -> integer action.
        """

        from_square = move.from_square
        to_square = move.to_square

        base = from_square * 64 + to_square

        if move.promotion is None:
            return base

        promotion_index = cls.PROMOTION_PIECES[move.promotion]

        return (
            cls.PROMOTION_OFFSET
            + base * 4
            + promotion_index
        )

    @classmethod
    def action_to_move(cls, action):
        """
        Convert integer action -> chess.Move.
        """

        action = int(action)

        if action < 0 or action >= cls.ACTION_SIZE:
            return None

        # Normal move
        if action < cls.PROMOTION_OFFSET:

            from_square = action // 64
            to_square = action % 64

            return chess.Move(
                from_square,
                to_square,
            )

        # Promotion move
        promotion_action = (
            action - cls.PROMOTION_OFFSET
        )

        promotion_index = promotion_action % 4

        base = promotion_action // 4

        from_square = base // 64
        to_square = base % 64

        promotion_piece = cls.REVERSE_PROMOTION[
            promotion_index
        ]

        return chess.Move(
            from_square,
            to_square,
            promotion=promotion_piece,
        )

    # ---------------------------------------------------------
    # Legal Action Mask
    # ---------------------------------------------------------

    def legal_action_mask(self):
        """
        Return a boolean mask of shape (ACTION_SIZE,).

        True  = legal action
        False = illegal action
        """

        mask = np.zeros(
            self.ACTION_SIZE,
            dtype=np.bool_,
        )

        for move in self.board.legal_moves:

            action = self.move_to_action(move)

            mask[action] = True

        return mask

    def legal_actions(self):
        """
        Return integer IDs for all legal moves.
        """

        return np.flatnonzero(
            self.legal_action_mask()
        )

    def legal_moves(self):
        """
        Return legal chess.Move objects.
        """

        return list(self.board.legal_moves)

    def legal_moves_uci(self):
        """
        Return legal moves in UCI notation.
        """

        return [
            move.uci()
            for move in self.board.legal_moves
        ]

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def get_fen(self):
        return self.board.fen()

    def render(self):
        print(self.board)
        print()
        print("FEN:", self.board.fen())
        print(
            "Turn:",
            "White"
            if self.board.turn == chess.WHITE
            else "Black",
        )

    def close(self):
        pass


# =============================================================
# TESTS
# =============================================================

def test_start_position():

    print("\n[TEST 1] Starting position")

    env = ChessEnvV3()

    env.reset()

    legal_moves = env.legal_moves()
    mask = env.legal_action_mask()

    assert len(legal_moves) == 20
    assert mask.sum() == 20

    print("Legal moves:", len(legal_moves))
    print("Legal actions:", int(mask.sum()))
    print("PASS")

    env.close()


def test_move_encoding():

    print("\n[TEST 2] Move encoding")

    test_moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("e1g1"),
        chess.Move.from_uci("e7e8q"),
    ]

    for move in test_moves:

        action = ChessEnvV3.move_to_action(move)

        decoded = ChessEnvV3.action_to_move(action)

        assert decoded == move

        print(
            move.uci(),
            "->",
            action,
            "->",
            decoded.uci(),
        )

    print("PASS")


def test_no_collision():

    print("\n[TEST 3] Action collision test")

    moves = []

    for from_square in range(64):
        for to_square in range(64):

            if from_square == to_square:
                continue

            move = chess.Move(
                from_square,
                to_square,
            )

            action = ChessEnvV3.move_to_action(move)

            moves.append(action)

    assert len(moves) == len(set(moves))

    print("Unique actions:", len(set(moves)))
    print("PASS")


def test_execute_legal_action():

    print("\n[TEST 4] Execute legal action")

    env = ChessEnvV3()

    env.reset()

    move = chess.Move.from_uci("e2e4")

    action = ChessEnvV3.move_to_action(move)

    assert action in env.legal_actions()

    _, reward, terminated, truncated, info = env.step(
        action
    )

    assert info["move"] == "e2e4"
    assert not terminated
    assert not truncated

    print("Move:", info["move"])
    print("SAN:", info["san"])
    print("Reward:", reward)
    print("PASS")

    env.close()


def test_invalid_action():

    print("\n[TEST 5] Invalid action handling")

    env = ChessEnvV3()

    env.reset()

    # e7e5 is a BLACK move and therefore illegal
    # in the initial WHITE position.
    action = ChessEnvV3.move_to_action(
        chess.Move.from_uci("e7e5")
    )

    assert action not in env.legal_actions()

    _, reward, terminated, truncated, info = env.step(
        action
    )

    assert info["invalid_action"] is True
    assert reward == -0.1
    assert not terminated
    assert not truncated

    print("Invalid action rejected correctly")
    print("Reward:", reward)
    print("PASS")

    env.close()


def test_promotion():

    print("\n[TEST 6] Promotion encoding")

    board = chess.Board(
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
    )

    env = ChessEnvV3()
    env.board = board

    move = chess.Move.from_uci("a7a8q")

    assert move in env.legal_moves()

    action = env.move_to_action(move)
    decoded = env.action_to_move(action)

    assert decoded == move

    print(
        "Promotion:",
        move.uci(),
        "->",
        action,
        "->",
        decoded.uci(),
    )

    print("PASS")

    env.close()


if __name__ == "__main__":

    print("=" * 70)
    print("CHESS ENVIRONMENT V2.3 TEST SUITE")
    print("=" * 70)

    print("\nAction space size:")
    print(ChessEnvV3.ACTION_SIZE)

    test_start_position()
    test_move_encoding()
    test_no_collision()
    test_execute_legal_action()
    test_invalid_action()
    test_promotion()

    print("\n" + "=" * 70)
    print("ALL V2.3 TESTS PASSED")
    print("=" * 70)
