from pprint import pprint

import numpy as np
from numba import njit
import numba

from .HiveConstants import *
from .HiveConstants import _decode_action, _encode_action, _np_all_axis1
# from .SmallworldMaps import *
# from .SmallworldDisplay import print_board, print_valids, move_to_str

############################## BOARD DESCRIPTION ##############################
#
#  grid tutorial https://www.redblobgames.com/grids/agons/#conversions
#
# Board sufficient for N pieces must include agon of N radia (ex. 3)
# It means the diameter is 2N-1 (ex. 5)
# It can be stored in array with 4N-3 (ex. 9) rows and N (ex. 3) columns
# White start in position (0, 0)
#
#   1     2     3  for odd rows
#      1     2     for even rows (Nth is excessive)
#  __    __    __
# /  \__/00\__/  \ _1 odd row
# \__/00\__/00\__/ _2 even row
# /00\__/00\__/00\ _3
# \__/00\__/00\__/ _4
# /00\__/wG\__/00\ _5
# \__/00\__/00\__/ _6
# /00\__/00\__/00\ _7
# \__/00\__/00\__/ _8
# /  \__/00\__/  \ _9
# \__/  \__/  \__/

#  \ / \ / \ / \ / \ /
#   |   |   |   |   |
#  / \ / \ / \ / \ / \
# |   |   |   |   |   |
#  \ / \ / \ / \ / \ /
#   |   |   |   |   |
#  / \ / \ / \ / \ / \
# |   |   |   |   |   |
#  \ / \ / \ / \ / \ /
#   |   |   |   |   |
#  / \ / \ / \ / \ / \
# |   |   |   |   |   |


# State is 2*PLAYER_PIECES_COUNT*2 — QR axial coordinates of all pieces
# 0,0 means player hand

# ACTION is bitfield
# 	field B: which bug used (PLAYER_PIECES_COUNT)
#	field Q: Q axial coord (BOARD_SIZE)
#	field R: Q axial coord (BOARD_SIZE)
#	W Q R

@njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (2*PLAYER_PIECES_COUNT + 1,2)

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return PLAYER_PIECES_COUNT*BOARD_SIZE*BOARD_SIZE + 1 # Plus 1 is for PASS

spec = [
	('state'      , numba.int8[:,:]),
	# ('board'      , numba.int8[:,:]),
	('pieces'     , numba.bool_[:,:]),
	('perimeter'  , numba.bool_[:,:]),
	('round_num'  , numba.int8[:]),
]
@numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros((2*PLAYER_PIECES_COUNT + 1,2), dtype=np.int8) # QR coordinates
		self.init_game()

	def get_score(self, player):
		# TODO Number of bugs around queen
		# self._get_bug_position(-1 * player * QUEEN)
		return 0

	def init_game(self):
		self.copy_state(np.zeros((2*PLAYER_PIECES_COUNT + 1,2), dtype=np.int8), copy_or_not=False)

		self.pieces = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		self.perimeter = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		self.round_num = np.zeros(2, dtype=np.int8)

	def get_state(self):
		return self.state

	def valid_moves(self, player):
		actions = np.zeros(action_size(), dtype=np.bool_)
		player_pieces = self._get_player_pieces(player)
		# First move
		if self.round_num[0] == 0:
			for piece in player_pieces:
				actions[_encode_action(piece, BOARD_SIZE//2, BOARD_SIZE//2)] = True
		# Second move
		elif self.round_num[0] == 1:
			for piece in player_pieces:
				actions[_encode_action(piece, BOARD_SIZE//2 + 1, BOARD_SIZE//2)] = True
		else:
			if self.round_num[0] == 6 and self._piece_in_hand(player_pieces[0]):
				player_pieces = np.array([player_pieces[0]])
			if self.round_num[0] == 7 and self._piece_in_hand(player_pieces[0]):
				player_pieces = np.array([player_pieces[0]])
			for piece in player_pieces:
				# Pieces on boards should not reject one_hive_rule
				if self._piece_in_play(piece) and not self._check_one_hive_rule(piece):
					continue
				# Pieces in hand to perimeter
				if self._piece_in_hand(piece):
					# TODO Remove union because perimeter and pieces should not intersect
					rows, cols = np.where(np.logical_and(self.perimeter, self.pieces == False))
					for r, c in zip(rows, cols):
						actions[_encode_action(piece, r, c)] = True
				# Piece in play somewhere
				else:
					for s in self._get_surroundings(self.state[piece][0], self.state[piece][1]):
						if self.pieces[s[0], s[1]]:
							continue
						count = 0
						for s1 in self._get_surroundings(s[0], s[1]):
							count += self.pieces[s1[0], s1[1]]
						if count >= 2 and self.state[piece][0] != s[0] and self.state[piece][1] != s[1]:
							actions[_encode_action(piece, s[0], s[1])] = True
		if sum(actions) == 0:
			actions[PLAYER_PIECES_COUNT*BOARD_SIZE*BOARD_SIZE] = True
		return actions

	def make_move(self, move, player, random_seed):
		self.round_num[0] += 1
		if move == PLAYER_PIECES_COUNT*BOARD_SIZE*BOARD_SIZE:
			return self._get_opponent(player)
		#
		# print(f'-'*11)
		# for r in range(BOARD_SIZE):
		# 	l = ''
		# 	for q in range(BOARD_SIZE):
		# 		if self.perimeter[q, r]:
		# 			l += f'+ '
		# 		else:
		# 			l += f'. '
		# 	print(r * ' ' + l)
		piece, new_q, new_r = _decode_action(move)
		q, r = self.state[piece + player * PLAYER_PIECES_COUNT]
		self.state[piece + player * PLAYER_PIECES_COUNT] = [new_q, new_r]
		# self.board[new_q,new_r] = piece
		self.pieces[q, r] = False
		self.pieces[new_q, new_r] = True
		if not (q == 0 and r == 0):
			self.perimeter[q, r] = True
			for s in self._get_surroundings(q, r):
				if self.pieces[s[0], s[1]]:
					continue
				count = 0
				for s1 in self._get_surroundings(s[0], s[1]):
					count += not self.pieces[s1[0], s1[1]]
				if count == 0:
					self.perimeter[s[0], s[1]] = False
		self.perimeter[new_q, new_r] = False
		for s in self._get_surroundings(new_q, new_r):
			if self.pieces[s[0], s[1]]:
				continue
			self.perimeter[s[0], s[1]] = True
		return self._get_opponent(player)

	def check_end_game(self, next_player):
		if self.round_num[0] > 100:
			return np.array([0.1, 0.1], dtype=np.float32)
		next_player_queen = self.state[self._get_player_pieces(next_player)[0]]
		occupied = 0
		for queen_surroundings in self._get_surroundings(next_player_queen[0], next_player_queen[1]):
			if self.pieces[queen_surroundings[0], queen_surroundings[1]]:
				occupied += 1
		if occupied == 6:
			if next_player == 0:
				return np.array([-1, 1], dtype=np.float32)
			else:
				return np.array([1, -1], dtype=np.float32)
		return np.array([0, 0], dtype=np.float32)

	def swap_players(self, nb_swaps):
		if nb_swaps != 1:
			return
		# Pieces exchange
		new_state = self.state.copy()
		for i in range(PLAYER_PIECES_COUNT):
			new_state[i] = self.state[PLAYER_PIECES_COUNT + i]
			new_state[PLAYER_PIECES_COUNT + i] = self.state[i]
		self.state = new_state

	def get_symmetries(self, policy, valid_actions):
		symmetries = [(self.state.copy(), policy.copy(), valid_actions.copy())]
		# TODO Implement Hive
		# state_backup = self.state.copy()
		#
		# # Rotate 90°, 180°, 270°
		# def _apply_permutation(permutation, array, array2):
		# 	array_copy, array2_copy = array.copy(), array2.copy()
		# 	for i, new_i in enumerate(permutation):
		# 		array_copy[new_i], array2_copy[new_i] = array[i], array2[i]
		# 	return array_copy, array2_copy
		#
		# def _apply_permutation_gods(permutation_gods, gods_power):
		# 	offset = 64+1
		# 	for i in range(2*NB_GODS):
		# 		if i%NB_GODS == ARTEMIS or i%NB_GODS == DEMETER:
		# 			if gods_power.flat[i] < offset:
		# 				continue
		# 			gods_power.flat[i] = permutation_gods[gods_power.flat[i] - offset]
		#
		# rotated_policy, rotated_actions = policy, valid_actions
		# for i in range(3):
		# 	self.workers[:,:] = np.rot90(self.workers)
		# 	self.levels[:,:] = np.rot90(self.levels)
		# 	rotated_policy, rotated_actions = _apply_permutation(rotation, rotated_policy, rotated_actions)
		# 	_apply_permutation_gods(rotation_gods, self.gods_power)
		# 	symmetries.append((self.state.copy(), rotated_policy.copy(), rotated_actions.copy()))
		# self.state[:,:,:] = state_backup.copy()
		#
		# # Mirror horizontally, vertically
		# self.workers[:,:] = np.fliplr(self.workers)
		# self.levels[:,:] = np.fliplr(self.levels)
		# _apply_permutation_gods(flipLR_gods, self.gods_power)
		# flipped_policy, flipped_actions = _apply_permutation(flipLR, policy, valid_actions)
		# symmetries.append((self.state.copy(), flipped_policy, flipped_actions))
		# self.state[:,:,:] = state_backup.copy()
		#
		# self.workers[:,:] = np.flipud(self.workers).copy()
		# self.levels[:,:] = np.flipud(self.levels).copy()
		# _apply_permutation_gods(flipUD_gods, self.gods_power)
		# flipped_policy, flipped_actions = _apply_permutation(flipUD, policy, valid_actions)
		# symmetries.append((self.state.copy(), flipped_policy, flipped_actions))
		# self.state[:,:,:] = state_backup.copy()
		#
		# if INIT_METHOD == 2 and np.abs(self.workers).sum() != 6:
		# 	return symmetries # workers not all set, stopping here
		#
		# # Permute worker 1 and 2
		# def _swap_workers(array, half_size):
		# 	array_copy = array.copy()
		# 	array_copy[:half_size], array_copy[half_size:] = array[half_size:], array[:half_size]
		# 	return array_copy
		#
		# def _swap_workers_gods(gods_power, player):
		# 	offset = 64+1
		# 	for i in range(NB_GODS*player, NB_GODS*(player+1)):
		# 		if i%NB_GODS == ARTEMIS or i%NB_GODS == DEMETER or i%NB_GODS == ATHENA:
		# 			if gods_power.flat[i] < offset:
		# 				continue
		# 			gods_power.flat[i] = (gods_power.flat[i]-offset + 9) % 18 + offset
		#
		# w1, w2 = self._get_worker_position(1), self._get_worker_position(2)
		# self.workers[w1], self.workers[w2] = 2, 1
		# _swap_workers_gods(self.gods_power, 0)
		# swapped_policy = _swap_workers(policy, action_size()//2)
		# swapped_actions = _swap_workers(valid_actions, action_size()//2)
		# symmetries.append((self.state.copy(), swapped_policy, swapped_actions))
		# self.state[:,:,:] = state_backup.copy()
		#
		# # Permute worker -1 and -2
		# wm1, wm2 = self._get_worker_position(-1), self._get_worker_position(-2)
		# self.workers[wm1], self.workers[wm2] = -2, -1
		# _swap_workers_gods(self.gods_power, 1)
		# symmetries.append((self.state.copy(), policy.copy(), valid_actions.copy()))
		# self.state[:,:,:] = state_backup.copy()

		return symmetries

	def get_round(self):
		return self.round_num[0]

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.pieces = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		for qr in self.state[:-1]:
			self.pieces[qr[0], qr[1]] = True
		self.perimeter = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		for i, qr in enumerate(self.state[:-1]):
			if self._piece_in_play(i):
				for s in self._get_surroundings(qr[0], qr[1]):
					if not self.pieces[s[0], s[1]]:
						self.perimeter[s[0], s[1]] = True
		self.round_num = self.state[-1,:]

	def _get_bug_position(self, searched_bug):
		for i in np.ndindex(BOARD_SIZE, BOARD_SIZE):
			if self.state[i] == searched_bug:
				return i
		print(f'Should not happen gwp {searched_bug}')
		return (-1, -1)

	def _get_surroundings(self, q, r):
		result = np.array([[q, r-1], [q+1, r-1],
						   [q-1, r], [q+1, r],
						   [q-1, r+1], [q, r+1]], dtype=np.int8)
		result = result[_np_all_axis1(result >= 0)]
		result = result[_np_all_axis1(result < BOARD_SIZE)]
		return result

	# def _get_spawns(self, player):
	# 	self.state[[self._get_player_pieces(player)]]


	def _get_player_pieces(self, player):
		return player * PLAYER_PIECES_COUNT + np.arange(PLAYER_PIECES_COUNT)

	# players are 0 and 1
	def _get_opponent(self, player):
		return 1 - player

	def _piece_in_play(self, piece):
		return not self._piece_in_hand(piece)

	def _piece_in_hand(self, piece):
		return self.state[piece][0] == 0 and self.state[piece][1] == 0

	def _check_one_hive_rule(self, piece):
		q, r = self.state[piece]
		self.pieces[q,r] = False
		stack = [self.state[piece]]
		stack.pop()
		seen = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		# TODO if 1 neighboor = can move
		for s in self._get_surroundings(q,r):
			if self.pieces[s[0], s[1]]:
				stack += [s]
				seen[s[0], s[1]] = True
				break
		while len(stack) > 0:
			i = stack.pop()
			# for j, piece in enumerate(self.state):
			# 	if np.all(piece == i):
			# 		print(j)
			for s in self._get_surroundings(i[0], i[1]):
				if self.pieces[s[0], s[1]] and not seen[s[0], s[1]]:
					stack += [s]
					seen[s[0], s[1]] = True
		self.pieces[q,r] = True
		if seen.sum() + 1 == self.pieces.sum():
			return True
		return False