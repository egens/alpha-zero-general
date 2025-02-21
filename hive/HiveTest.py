import base64
import zlib

from numpy import frombuffer, int8
import numpy as np

from hive.HiveConstants import _decode_action, _encode_action, PLAYER_PIECES_COUNT
from hive.HiveDisplay import _get_char, _print_moves, _print_flags
from hive.HiveGame import HiveGame
from hive.HivePlayers import GreedyPlayer, RandomPlayer

if __name__ == '__main__':
    for to in range(PLAYER_PIECES_COUNT):
        for p in range(PLAYER_PIECES_COUNT):
            for d in range(6):
                a = _encode_action(p, to, d)
                p1, to1, d1, f = _decode_action(a)
                assert to == to1
                assert p == p1
                assert d == d1

    game  = HiveGame()
    board = game.getInitBoard()
    state = "BcEBDgAgCAMxE0F2G/9/r63GQrm2r4pSZ5jARlO0Y8TCi1dn6wM="
    data = zlib.decompress(base64.b64decode(state), wbits=-15)
    board = frombuffer(data[:-3], dtype=int8).reshape(board.shape)
    game.getCanonicalForm(board, 0)
    _print_flags(game.board)
    game.printBoard(board)
    p = GreedyPlayer(game)
    print(game.getCanonicalForm(board,0))
    # p = RandomPlayer(game)
    print(p.play(board, 0))
    game.printBoard(board)

