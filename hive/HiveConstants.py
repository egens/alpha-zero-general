from numba import njit
import numpy as np

BOARD_SIZE = 15

PIECE_TYPES_NUM = 2
PLAYER_PIECES_COUNT = 7

QUEEN = 0
ANT_1 = 1
ANT_2 = 2
ANT_3 = 3
ANT_4 = 4
ANT_5 = 5
ANT_6 = 6
ANT_7 = 7
ANT_8 = 8
ANT_9 = 9

ANTS = {ANT_1, ANT_2, ANT_3, ANT_4, ANT_5, ANT_6, ANT_7, ANT_8, ANT_9}
# BEETLE
# SPIDER
# GRASSHOPPER
# LADYBUG
# MOSQITO
# PILLBUG

@njit(cache=True, fastmath=True, nogil=True)
def _decode_action(action):
	new_q, action_ = divmod(action, BOARD_SIZE*PLAYER_PIECES_COUNT)
	new_r, piece = divmod(action_, PLAYER_PIECES_COUNT)
	return piece, new_q, new_r

@njit(cache=True, fastmath=True, nogil=True)
def _encode_action(piece, new_q, new_r):
	action = BOARD_SIZE*PLAYER_PIECES_COUNT*new_q + PLAYER_PIECES_COUNT*new_r + piece
	return action

@njit(cache=True)
def _np_all_axis1(x):
	"""Numba compatible version of np.all(x, axis=1)."""
	out = np.ones(x.shape[0], dtype=np.bool)
	for i in range(x.shape[1]):
		out = np.logical_and(out, x[:, i])
	return out

# 9 10 0 2193
# 9 15 19 1925
# 9 17 19 1809
# P1 decided to move 20 to 13 15.0
# c 1
# p 1
# 2193
# (10, 9, 13)
if __name__ == '__main__':
	print(_decode_action(1137))