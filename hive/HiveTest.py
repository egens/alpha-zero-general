import base64
import zlib
from random import random

from numpy import frombuffer, int8
import numpy as np

from hive.HiveConstants import _decode_action, _encode_action, PLAYER_PIECES_COUNT
from hive.HiveDisplay import _get_char, _print_moves, _print_flags, move_to_str
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
    state = "+/8fAlgY/hMEjAyMjAwA"
    # P1 decided to move S2 to (1430, 21, 20, 1, 0) S1-
    data = zlib.decompress(base64.b64decode(state), wbits=-15)
    board = frombuffer(data[:-3], dtype=int8).reshape(board.shape)
    curPlayer = 0
    canonical_board = game.getCanonicalForm(board, curPlayer)
    game.printBoard(board)
    valids = game.getValidMoves(board, player=0)
    print('!!!', np.where(valids)[0])
    valids = game.getValidMoves(board, player=0)
    print('!!!', np.where(valids)[0])
    b,p = game.getNextState(board, 1, 1430)
    game.printBoard(b)
    # action = random.choices(range(game.getActionSize()), weights=valids.astype(int), k=1)[0]

    # for m in np.where(game.getValidMoves(canonical_board, 0))[0]:
    #     print(move_to_str(m, 0))

    # action = players[curPlayer](canonical_board, it)

    rp = RandomPlayer(game)
    rp.play(canonical_board, 4)