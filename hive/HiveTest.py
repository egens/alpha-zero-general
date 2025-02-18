import base64
import zlib

from numpy import frombuffer, int8
import numpy as np

from hive.HiveConstants import _decode_action
from hive.HiveDisplay import _get_char
from hive.HiveGame import HiveGame

if __name__ == '__main__':
    game  = HiveGame()
    board = game.getInitBoard()
    state = "Dci3AQAwCAMwLwEXruD/I4NGkT01itsUUyGbyqvYzKVx5AUWHw=="
    data = zlib.decompress(base64.b64decode(state), wbits=-15)
    board = frombuffer(data[:-3], dtype=int8).reshape(board.shape)
    game.printBoard(board)
    for a, valid in enumerate(game.getValidMoves(board, 0)):
        if valid:
            piece, new_q, new_r = _decode_action(a)
            print(a, _get_char(piece), new_q, new_r)