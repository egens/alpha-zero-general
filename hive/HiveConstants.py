from numba import njit
import numpy as np

BOARD_SIZE = 15

PIECE_TYPES_NUM = 2
PLAYER_PIECES_COUNT = 10
PASS_ACTION = PLAYER_PIECES_COUNT*BOARD_SIZE*BOARD_SIZE

QUEEN = 0
ANT_1 = 1
ANT_2 = 2
ANT_3 = 3
BEETLE_1 = 4
BEETLE_2 = 5
BEETLE_3 = 6
GRASSHOPPER_1 = 7
GRASSHOPPER_2 = 8
GRASSHOPPER_3 = 9

@njit(cache=True, fastmath=True, nogil=True)
def _is_queen(p):
	n = p % PLAYER_PIECES_COUNT
	return n == 0

@njit(cache=True, fastmath=True, nogil=True)
def _is_ant(p):
	n = p % PLAYER_PIECES_COUNT
	return n == 1 or n == 2 or n == 3

@njit(cache=True, fastmath=True, nogil=True)
def _is_beetle(p):
	n = p % PLAYER_PIECES_COUNT
	return n == 4 or n == 5 or n == 6

@njit(cache=True, fastmath=True, nogil=True)
def _is_grasshopper(p):
	n = p % PLAYER_PIECES_COUNT
	return n == 7 or n == 8 or n == 9

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

# https://stackoverflow.com/a/65622029/1488538
@njit(cache=True)
def _np_all_axis1(x):
	"""Numba compatible version of np.all(x, axis=1)."""
	out = np.ones(x.shape[0], dtype=np.bool)
	for i in range(x.shape[1]):
		out = np.logical_and(out, x[:, i])
	return out

@njit(cache=True)
def _np_all_axis0(x):
	"""Numba compatible version of np.all(x, axis=0)."""
	out = np.ones(x.shape[1], dtype=np.bool)
	for i in range(x.shape[0]):
		out = np.logical_and(out, x[i, :])
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