import chess
import numpy as np

from chess_env_v4 import ChessEnvV4


class MaskedChessEnv(ChessEnvV4):
    """
    ChessEnvV4 with a MaskablePPO-compatible action mask.
    """

    def action_masks(self):
        """
        sb3-contrib expects:
            True  = action is valid
            False = action is invalid
        """
        return self.legal_action_mask()
