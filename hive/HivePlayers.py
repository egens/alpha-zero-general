import numpy as np
import random

from .HiveDisplay import print_board, move_to_str, _get_char
from .HiveConstants import _encode_action, _decode_action

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, nb_moves):
        valids = self.game.getValidMoves(board, player=0)
        action = random.choices(range(self.game.getActionSize()), weights=valids.astype(int), k=1)[0]
        return action


class HumanPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, nb_moves):
        # print_board(self.game.board)
        valid = self.game.getValidMoves(board, 0)
        while True:
            input_move = input('Select move ')
            try:
                a = int(input_move)
                if not valid[a]:
                    raise Exception('')
                break
            except:
                print('Invalid move:', input_move)
        return a

class GreedyPlayer():
    pass
