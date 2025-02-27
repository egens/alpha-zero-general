from numba import njit
import numpy as np

BOARD_SIZE = 16

PIECE_TYPES_NUM = 2
PLAYER_PIECES_COUNT = 11
PASS_ACTION = PLAYER_PIECES_COUNT*PLAYER_PIECES_COUNT*6*2
MOVES_COUNT = PASS_ACTION + 1

QUEEN = 0
ANT_1 = 1
ANT_2 = 2
ANT_3 = 3
GRASSHOPPER_1 = 4
GRASSHOPPER_2 = 5
GRASSHOPPER_3 = 6
BEETLE_1 = 7
BEETLE_2 = 8

SPIDER_1 = 9
SPIDER_2 = 10

BEETLES = np.array([BEETLE_1, BEETLE_2, BEETLE_1+PLAYER_PIECES_COUNT, BEETLE_2+PLAYER_PIECES_COUNT], dtype=np.int8)

# Directions
#   5 \ / 0
#  4 - P - 1
#   3 / \ 2
DIRECTIONS = np.array([[1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1]], dtype=np.int8)
DIRECTIONS_STRING = '/-\\/-\\'

@njit(cache=True, fastmath=True, nogil=True)
def _is_queen(p):
	n = p % PLAYER_PIECES_COUNT
	return n == 0

@njit(cache=True, fastmath=True, nogil=True)
def _is_ant(p):
	n = p % PLAYER_PIECES_COUNT
	return n == 1 or n == 2 or n == 3

@njit(cache=True, fastmath=True, nogil=True)
def _is_grasshopper(p):
	n = p % PLAYER_PIECES_COUNT
	return n == 4 or n == 5 or n == 6

@njit(cache=True, fastmath=True, nogil=True)
def _is_beetle(p):
	n = p % PLAYER_PIECES_COUNT
	return n == 7 or n == 8

@njit(cache=True, fastmath=True, nogil=True)
def _is_spider(p):
	n = p % PLAYER_PIECES_COUNT
	return n == 9 or n == 10

# SPIDER
# GRASSHOPPER
# LADYBUG
# MOSQITO
# PILLBUG

# Move to piece actions
@njit(cache=True, fastmath=True, nogil=True)
def _decode_action(action, player=None):
	piece, action_ = divmod(action, PLAYER_PIECES_COUNT*6*2)
	to_piece, action_ = divmod(action_, 6*2)
	direction, is_opponent_piece = divmod(action_, 2)

	if player == 1:
		piece += PLAYER_PIECES_COUNT
	if is_opponent_piece and player == 0:
		to_piece = to_piece + PLAYER_PIECES_COUNT
	if not is_opponent_piece and player == 1:
		to_piece = to_piece + PLAYER_PIECES_COUNT

	return piece, to_piece, direction, is_opponent_piece

@njit(cache=True, fastmath=True, nogil=True)
def _encode_action(piece, to_piece, direction):
	piece_player = piece < PLAYER_PIECES_COUNT
	to_piece_player = to_piece < PLAYER_PIECES_COUNT
	is_opponent_piece = piece_player != to_piece_player
	piece = piece % PLAYER_PIECES_COUNT
	to_piece = to_piece % PLAYER_PIECES_COUNT
	action = PLAYER_PIECES_COUNT*6*2*piece + 6*2*to_piece + 2*direction + is_opponent_piece
	return action

# Q-R actions
# @njit(cache=True, fastmath=True, nogil=True)
# def _decode_action(action):
# 	new_q, action_ = divmod(action, BOARD_SIZE*PLAYER_PIECES_COUNT)
# 	new_r, piece = divmod(action_, PLAYER_PIECES_COUNT)
# 	return piece, new_q, new_r
#
# @njit(cache=True, fastmath=True, nogil=True)
# def _encode_action(piece, new_q, new_r):
# 	action = BOARD_SIZE*PLAYER_PIECES_COUNT*new_q + PLAYER_PIECES_COUNT*new_r + piece
# 	return action

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