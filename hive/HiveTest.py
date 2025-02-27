import base64
import zlib

from numpy import frombuffer, int8
import numpy as np

from hive.HiveConstants import _decode_action, _encode_action, PLAYER_PIECES_COUNT
from hive.HiveDisplay import _get_char, _print_moves, _print_flags, move_to_str
from hive.HiveGame import HiveGame
from hive.HivePlayers import GreedyPlayer, RandomPlayer

if __name__ == '__main__':
    game  = HiveGame()
    for to in range(PLAYER_PIECES_COUNT):
        for p in range(PLAYER_PIECES_COUNT):
            for d in range(6):
                a = _encode_action(p, to, d)
                p1, to1, d1, f = _decode_action(a)
                assert to == to1
                assert p == p1
                assert d == d1


    board = game.getInitBoard()
    state = "42RjAAMONjZ2NjCbg52BgZ2dAQmwc0Dw//8i/3kZGHkZAA=="
    data = zlib.decompress(base64.b64decode(state), wbits=-15)
    board = frombuffer(data[:-3], dtype=int8).reshape(board.shape)
    # print(game.getCanonicalForm(board, 1))
    for m in np.where(game.getValidMoves(game.getCanonicalForm(board, 1), 0))[0]:
        # print(m)
        print(move_to_str(m, 0))
    print(game.getCanonicalForm(board, 0))
    _print_flags(game.board)
    game.printBoard(board)
    p = GreedyPlayer(game)
    print(game.getCanonicalForm(board,0))
    # p = RandomPlayer(game)
    print(p.play(board, 0))
    game.printBoard(board)

