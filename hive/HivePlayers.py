import numpy as np
import random

from .HiveDisplay import print_board, move_to_str, _get_char, _print_moves
from .HiveConstants import _encode_action, _decode_action, PLAYER_PIECES_COUNT, DIRECTIONS


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
        valids = self.game.getValidMoves(board, 0)
        while True:
            input_move = input('Select move ')
            if input_move == '':
                return random.choices(range(self.game.getActionSize()), weights=valids.astype(int), k=1)[0]
            try:
                a = int(input_move)
                if not valids[a]:
                    raise Exception('')
                break
            except:
                print('Invalid move:', input_move)
        return a

class GreedyPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, nb_moves):
        print('GREEDY')
        valids = self.game.getValidMoves(board, player=0)
        # Do not touch pieces from the opponent queen
        for a in np.where(valids)[0]:
            piece, to_piece, direction, is_opponent_piece = _decode_action(a)
            q, r = self.game.board.state[piece]
            if self.game.board._near_opponent_queen(0, q, r):
                valids[a] = False
        if sum(valids) == 0: # If all valid moves was filtered return them
            valids = self.game.getValidMoves(board, player=0)

        # If there is move to opponent queen make it
        for a in np.where(valids)[0]:
            piece, to_piece, direction, is_opponent_piece = _decode_action(a)
            player = 0
            if player == 1:
                piece += PLAYER_PIECES_COUNT
            if is_opponent_piece and player == 0:
                to_piece = to_piece + PLAYER_PIECES_COUNT
            if not is_opponent_piece and player == 1:
                to_piece = to_piece + PLAYER_PIECES_COUNT
            new_q, new_r = self.game.board.state[to_piece] + DIRECTIONS[direction]
            if self.game.board._near_opponent_queen(0, new_q, new_r):
                print('NEAR QUEEN')
                return a

        # If there is move to be closer to opponent queen make it
        for a in np.where(valids)[0]:
            piece, to_piece, direction, is_opponent_piece = _decode_action(a)
            player = 0
            if player == 1:
                piece += PLAYER_PIECES_COUNT
            if is_opponent_piece and player == 0:
                to_piece = to_piece + PLAYER_PIECES_COUNT
            if not is_opponent_piece and player == 1:
                to_piece = to_piece + PLAYER_PIECES_COUNT
            new_q, new_r = self.game.board.state[to_piece] + DIRECTIONS[direction]
            q, r = self.game.board.state[piece]
            if q + r == 0:
                continue
            if self.game.board._closer_to_opponent_queen(0, piece, new_q, new_r):
                print('CLOSER TO QUEEN')
                return a

        return random.choices(range(self.game.getActionSize()), weights=valids.astype(int), k=1)[0]
