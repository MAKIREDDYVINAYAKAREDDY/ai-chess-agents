import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ChessEnvV4(gym.Env):
    """
    V2.4 Chess RL environment with reward shaping.

    Observation:
        8x8x18

    Action:
        Fixed chess move encoding with 20,480 actions.

    Reward shaping:
        Checkmate        +10.0
        Win              +10.0
        Capture          +0.1 to +0.8
        Give check       +0.2
        Draw             +1.0
        Stalemate        +1.0
        Illegal move     -0.5
        Loss             -10.0
        Normal move       0.0
    """

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

    PIECE_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
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
            shape=(1152,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(
            self.ACTION_SIZE
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self.board = chess.Board()
        self.move_count = 0

        return self._get_observation(), {
            "fen": self.board.fen(),
            "turn": "white",
        }

    # =========================================================
    # STEP
    # =========================================================

    def step(self, action):

        action = int(action)

        move = self.action_to_move(action)

        # -----------------------------------------------------
        # Invalid action
        # -----------------------------------------------------

        if (
            move is None
            or move not in self.board.legal_moves
        ):
            return (
                self._get_observation(),
                -0.5,
                False,
                False,
                {
                    "invalid_action": True,
                    "action": action,
                    "fen": self.board.fen(),
                },
            )

        # -----------------------------------------------------
        # Store information BEFORE move
        # -----------------------------------------------------

        moving_color = self.board.turn

        captured_piece = None

        if self.board.is_capture(move):

            if self.board.is_en_passant(move):
                captured_piece = chess.PAWN
            else:
                captured_piece = self.board.piece_at(
                    move.to_square
                ).piece_type

        san = self.board.san(move)

        # -----------------------------------------------------
        # Make move
        # -----------------------------------------------------

        self.board.push(move)

        self.move_count += 1

        terminated = False
        truncated = False

        reward = 0.0

        # -----------------------------------------------------
        # Capture reward
        # -----------------------------------------------------

        capture_reward = 0.0

        if captured_piece is not None:

            value = self.PIECE_VALUES[
                captured_piece
            ]

            # Keep intermediate reward bounded.
            capture_reward = min(
                value * 0.1,
                0.8
            )

            reward += capture_reward

        # -----------------------------------------------------
        # Check reward
        # -----------------------------------------------------

        check_reward = 0.0

        if self.board.is_check():
            check_reward = 0.2
            reward += check_reward

        # -----------------------------------------------------
        # Terminal states
        # -----------------------------------------------------

        if self.board.is_checkmate():

            terminated = True

            reward += 10.0

            result = self.board.result()

        elif self.board.is_stalemate():

            terminated = True

            reward += 1.0

            result = "1/2-1/2"

        elif self.board.is_insufficient_material():

            terminated = True

            reward += 1.0

            result = "1/2-1/2"

        elif self.board.is_fivefold_repetition():

            terminated = True

            reward += 1.0

            result = "1/2-1/2"

        elif self.board.is_seventyfive_moves():

            terminated = True

            reward += 1.0

            result = "1/2-1/2"

        elif self.board.is_variant_loss():

            terminated = True

            reward -= 10.0

            result = self.board.result()

        elif self.board.is_variant_win():

            terminated = True

            reward += 10.0

            result = self.board.result()

        else:

            result = "*"

        # -----------------------------------------------------
        # Episode timeout
        # -----------------------------------------------------

        if (
            self.move_count >= self.max_moves
            and not terminated
        ):
            truncated = True

        # -----------------------------------------------------
        # Information
        # -----------------------------------------------------

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
            "capture_reward": capture_reward,
            "check_reward": check_reward,
            "legal_moves": len(
                list(self.board.legal_moves)
            ),
            "result": result,
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

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_observation(self):

        state = np.zeros(
            (8, 8, 18),
            dtype=np.float32
        )

        # Piece planes
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

        # En passant
        if self.board.ep_square is not None:

            square = self.board.ep_square

            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)

            state[row, col, 17] = 1.0

        return state.flatten()

    # =========================================================
    # ACTION ENCODING
    # =========================================================

    @classmethod
    def move_to_action(cls, move):

        from_square = move.from_square
        to_square = move.to_square

        base = (
            from_square * 64
            + to_square
        )

        if move.promotion is None:
            return base

        promotion_index = cls.PROMOTION_PIECES[
            move.promotion
        ]

        return (
            cls.PROMOTION_OFFSET
            + base * 4
            + promotion_index
        )

    @classmethod
    def action_to_move(cls, action):

        action = int(action)

        if (
            action < 0
            or action >= cls.ACTION_SIZE
        ):
            return None

        # Normal move
        if action < cls.PROMOTION_OFFSET:

            from_square = action // 64
            to_square = action % 64

            return chess.Move(
                from_square,
                to_square
            )

        # Promotion
        promotion_action = (
            action
            - cls.PROMOTION_OFFSET
        )

        promotion_index = (
            promotion_action % 4
        )

        base = promotion_action // 4

        from_square = base // 64
        to_square = base % 64

        promotion_piece = (
            cls.REVERSE_PROMOTION[
                promotion_index
            ]
        )

        return chess.Move(
            from_square,
            to_square,
            promotion=promotion_piece
        )

    # =========================================================
    # LEGAL ACTION MASK
    # =========================================================

    def legal_action_mask(self):

        mask = np.zeros(
            self.ACTION_SIZE,
            dtype=np.bool_
        )

        for move in self.board.legal_moves:

            action = self.move_to_action(move)

            mask[action] = True

        return mask

    def legal_actions(self):

        return np.flatnonzero(
            self.legal_action_mask()
        )

    def legal_moves(self):

        return list(
            self.board.legal_moves
        )

    # =========================================================
    # UTILITIES
    # =========================================================

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
            else "Black"
        )


# =============================================================
# TESTS
# =============================================================

def test_basic_rewards():

    print("\n[TEST 1] Basic move reward")

    env = ChessEnvV4()

    env.reset()

    move = chess.Move.from_uci(
        "e2e4"
    )

    action = env.move_to_action(move)

    _, reward, terminated, truncated, info = (
        env.step(action)
    )

    assert info["move"] == "e2e4"
    assert reward == 0.0
    assert not terminated
    assert not truncated

    print("Move:", info["san"])
    print("Reward:", reward)
    print("PASS")

    env.close()


def test_capture_reward():

    print("\n[TEST 2] Capture reward")

    env = ChessEnvV4()

    env.reset()

    # Position where white can capture black queen.
    env.board = chess.Board(
        "4k3/8/8/3q4/4Q3/8/8/4K3 w - - 0 1"
    )

    move = chess.Move.from_uci(
        "e4d5"
    )

    assert move in env.legal_moves()

    action = env.move_to_action(move)

    _, reward, _, _, info = env.step(
        action
    )

    assert info["capture_reward"] == 0.8
    assert reward >= 0.8

    print("Move:", info["san"])
    print(
        "Capture reward:",
        info["capture_reward"]
    )
    print("Total reward:", reward)
    print("PASS")

    env.close()


def test_check_reward():

    print("\n[TEST 3] Check reward")

    env = ChessEnvV4()

    env.reset()

    env.board = chess.Board(
        "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    )

    move = chess.Move.from_uci(
        "e2e8"
    )

    # e2e8 is not legal because black king occupies e8.
    # Use a legal checking move instead.
    move = chess.Move.from_uci(
        "e2e7"
    )

    assert move in env.legal_moves()

    action = env.move_to_action(move)

    _, reward, _, _, info = env.step(
        action
    )

    assert info["check_reward"] == 0.2

    print("Move:", info["san"])
    print(
        "Check reward:",
        info["check_reward"]
    )
    print("Total reward:", reward)
    print("PASS")

    env.close()


def test_checkmate_reward():

    print("\n[TEST 4] Checkmate reward")

    env = ChessEnvV4()

    env.reset()

    # Black king is trapped.
    env.board = chess.Board(
        "7k/5Q2/7K/8/8/8/8/8 w - - 0 1"
    )

    move = chess.Move.from_uci(
        "f7g7"
    )

    assert move in env.legal_moves()

    action = env.move_to_action(move)

    _, reward, terminated, _, info = (
        env.step(action)
    )

    assert terminated
    assert reward >= 10.0

    print("Move:", info["san"])
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("PASS")

    env.close()


def test_illegal_reward():

    print("\n[TEST 5] Illegal action penalty")

    env = ChessEnvV4()

    env.reset()

    illegal_move = chess.Move.from_uci(
        "e7e5"
    )

    action = env.move_to_action(
        illegal_move
    )

    _, reward, _, _, info = env.step(
        action
    )

    assert info["invalid_action"]
    assert reward == -0.5

    print("Reward:", reward)
    print("PASS")

    env.close()


def test_mask():

    print("\n[TEST 6] Legal-action mask")

    env = ChessEnvV4()

    env.reset()

    mask = env.legal_action_mask()

    assert mask.dtype == np.bool_
    assert mask.sum() == 20

    print(
        "Action space:",
        env.action_space.n
    )

    print(
        "Legal actions:",
        int(mask.sum())
    )

    print("PASS")

    env.close()


def test_encoding():

    print("\n[TEST 7] Encoding round-trip")

    moves = [
        "e2e4",
        "g1f3",
        "e1g1",
        "a7a8q",
        "a7a8r",
        "a7a8b",
        "a7a8n",
    ]

    for uci in moves:

        move = chess.Move.from_uci(uci)

        action = ChessEnvV4.move_to_action(
            move
        )

        decoded = ChessEnvV4.action_to_move(
            action
        )

        assert decoded == move

        print(
            uci,
            "->",
            action,
            "->",
            decoded.uci()
        )

    print("PASS")


if __name__ == "__main__":

    print("=" * 70)
    print("CHESS ENVIRONMENT V2.4 TEST SUITE")
    print("=" * 70)

    print(
        "\nAction space:",
        ChessEnvV4.ACTION_SIZE
    )

    test_basic_rewards()
    test_capture_reward()
    test_check_reward()
    test_checkmate_reward()
    test_illegal_reward()
    test_mask()
    test_encoding()

    print("\n" + "=" * 70)
    print("ALL V2.4 TESTS PASSED")
    print("=" * 70)
