import numba

from .HiveConstants import *
from .HiveConstants import _decode_action, _encode_action, _np_all_axis1, _is_queen, _is_beetle, _is_ant, \
	_is_grasshopper


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
# Round — first
# TODO Height of bugs and mosquitoes
# 0,0 means player hand

# ACTION is bitfield — last is pass
# 	field B: which bug used (PLAYER_PIECES_COUNT)
#	field Q: Q axial coord (BOARD_SIZE)
#	field R: R axial coord (BOARD_SIZE)
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
	('cutpoints'  , numba.bool_[:,:]),
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
		self.cutpoints = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		self.round_num = np.zeros(2, dtype=np.int8)

	def get_state(self):
		return self.state

	def valid_moves(self, player):
		# self._print_pieces()
		actions = np.zeros(action_size(), dtype=np.bool_)
		player_pieces = self._get_player_pieces(player)
		# First move white to the middle
		if self.round_num[0] == 0:
			for piece in player_pieces:
				actions[_encode_action(piece, BOARD_SIZE//2, BOARD_SIZE//2)] = True
		# Second move black to thr right of white
		elif self.round_num[0] == 1:
			for piece in player_pieces:
				actions[_encode_action(piece, BOARD_SIZE//2 + 1, BOARD_SIZE//2)] = True
		else:
			if self.round_num[0] == 6 and self._piece_in_hand(player_pieces[0]):
				player_pieces = np.array([player_pieces[0]])
			if self.round_num[0] == 7 and self._piece_in_hand(player_pieces[0]):
				player_pieces = np.array([player_pieces[0]])
			for piece in player_pieces:
				# Pieces in hand to perimeter
				if self._piece_in_hand(piece):
					rows, cols = np.where(self.perimeter)
					for r, c in zip(rows, cols):
						reject_action = False
						for s in self._get_surroundings(r, c):
							for p in self._get_player_pieces(self._get_opponent(player)):
								if self.state[p][0] == s[0] and self.state[p][1] == s[1]:
									reject_action = True
									continue
							if reject_action:
								continue
						if reject_action:
							continue
						actions[_encode_action(piece, r, c)] = True
				# Piece in play
				else:
					if self.cutpoints[self.state[piece][0], self.state[piece][1]]:
						continue
					# if self._beetle_above(piece):
					# 	continue
					if self._queen_in_hand(player):
						continue
					if _is_queen(piece) or _is_beetle(piece):
						surr = self._get_surroundings(self.state[piece][0], self.state[piece][1])
						for i in range(6):
							if self.pieces[surr[i][0], surr[i][1]]:
								continue
							nexti = (i + 1) % 6
							previ = (i - 1) % 6
							if self.pieces[surr[previ][0], surr[previ][1]] and not self.pieces[surr[nexti][0], surr[nexti][1]]:
								actions[_encode_action(piece, surr[i][0], surr[i][1])] = True
							if not self.pieces[surr[previ][0], surr[previ][1]] and self.pieces[surr[nexti][0], surr[nexti][1]]:
								actions[_encode_action(piece, surr[i][0], surr[i][1])] = True
					elif _is_ant(piece):
						rows, cols = np.where(self.perimeter)
						for r, c in zip(rows, cols):
							count = 0
							for s in self._get_surroundings(r, c):
								if self.pieces[s[0], s[1]]:
									count += 1
							if count >= 2:
								actions[_encode_action(piece, r, c)] = True
					elif _is_grasshopper(piece):
						coord = self.state[piece]
						for axis in [[0, -1], [1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0]]:
							if self.pieces[coord[0] + axis[0], coord[1] + axis[1]]:
								for i in range(1, BOARD_SIZE):
									if not self.pieces[coord[0] + i*axis[0], coord[1] + i*axis[1]]:
										actions[_encode_action(piece, coord[0] + i*axis[0], coord[1] + i*axis[1])] = True
										break

		if sum(actions) == 0:
			actions[PASS_ACTION] = True
		return actions

	def _get_ant_moves(self, piece):
		pass

	def make_move(self, move, player, random_seed):
		self.round_num[0] += 1
		if move != PASS_ACTION:
			piece, new_q, new_r = _decode_action(move)
			self.state[piece + player * PLAYER_PIECES_COUNT] = [new_q, new_r]
		return self._get_opponent(player)

	def check_end_game(self, next_player):
		# Ideally game should be over before 100 move
		if self.round_num[0] > 100:
			return np.array([0.1, 0.1], dtype=np.float32)
		cur_player_queen = self.state[self._get_player_pieces(self._get_opponent(next_player))[0]]
		cur_player_health = 6
		for queen_surroundings in self._get_surroundings(cur_player_queen[0], cur_player_queen[1]):
			if self.pieces[queen_surroundings[0], queen_surroundings[1]]:
				cur_player_health -= 1
		next_player_queen = self.state[self._get_player_pieces(next_player)[0]]
		next_player_health = 6
		for queen_surroundings in self._get_surroundings(next_player_queen[0], next_player_queen[1]):
			if self.pieces[queen_surroundings[0], queen_surroundings[1]]:
				next_player_health -= 1
		if cur_player_health == 0 and next_player_health == 0:
			np.array([0.1, 0.1], dtype=np.float32)
		if next_player_health == 0:
			if next_player == 0:
				return np.array([-1, 1], dtype=np.float32)
			else:
				return np.array([1, -1], dtype=np.float32)
		if cur_player_health == 0:
			if next_player == 0:
				return np.array([1, -1], dtype=np.float32)
			else:
				return np.array([-1, 1], dtype=np.float32)
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

		return symmetries

	def get_round(self):
		return self.round_num[0]

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.pieces = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		for i, qr in enumerate(self.state[:-1]):
			if self._piece_in_play(i):
				self.pieces[qr[0], qr[1]] = True
		self.perimeter = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		for i, qr in enumerate(self.state[:-1]):
			if self._piece_in_play(i):
				for s in self._get_surroundings(qr[0], qr[1]):
					if not self.pieces[s[0], s[1]]:
						self.perimeter[s[0], s[1]] = True
		self.round_num = self.state[-1,:]
		self.cutpoints = self._get_cutpoints()
		# self._print_flags()

	def _get_bug_position(self, searched_bug):
		for i in np.ndindex(BOARD_SIZE, BOARD_SIZE):
			if self.state[i] == searched_bug:
				return i
		print(f'Should not happen gwp {searched_bug}')
		return (-1, -1)

	def _get_surroundings(self, q, r):
		result = np.array([[q, r-1], [q+1, r-1],
						   [q+1, r], [q, r+1],
						   [q-1, r+1], [q-1, r]], dtype=np.int8)
		result = result[_np_all_axis1(result >= 0)]
		result = result[_np_all_axis1(result < BOARD_SIZE)]
		return result

	def _get_surrounding_pieces(self, q, r):
		idxs = np.zeros(6, dtype=np.int8)
		surrs = self._get_surroundings(q,r)
		count = 0
		for i, coord in enumerate(surrs):
			q1, r1 = coord
			if self.pieces[q1, r1]:
				idxs[count] = i
				count += 1
		return surrs[idxs[:count]]

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

	# Find pieces blocked by one hive rule
	# https://en.wikipedia.org/wiki/Biconnected_component
	# Iterative implementation for numba
	def _get_cutpoints(self):
		UNVISITED = 0
		VISITING = 1
		VISITED = 2

		status = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.int8)
		parent = np.zeros((BOARD_SIZE,BOARD_SIZE,2), dtype=np.int8) - 1
		depth = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.int8) - 1
		low = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.int8) - 1
		cutpoints = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		root_child_count = 0

		# Find one piece in play and start search from its pos
		stack = [self.state[0]]
		stack.pop()
		for p in range(2*PLAYER_PIECES_COUNT):
			if self._piece_in_play(p):
				stack = [self.state[p]]
				q, r = stack[-1]
				status[q, r] = VISITING
				break
		while len(stack) > 0:
			q, r = stack[-1]
			pq, pr = parent[q, r]
			current_is_root = pr == -1 and pr == -1
			if current_is_root:
				depth[q, r] = 0
				low[q, r] = 0
			else:
				depth[q, r] = depth[pq, pr] + 1
				low[q, r] = low[pq, pr] + 1

			surr_unvisited = 0
			for s in self._get_surrounding_pieces(q, r):
				q1, r1 = s
				if status[q1, r1] == UNVISITED:
					status[q1, r1] = VISITING
					stack += [s]
					surr_unvisited += 1
					parent[q1, r1] = [q, r]
					if current_is_root:
						root_child_count += 1

			if surr_unvisited == 0:
				status[q, r] = VISITED
				stack.pop()
				current_is_articulation = False
				for s in self._get_surrounding_pieces(q, r):
					q1, r1 = s
					if low[q1, r1] >= depth[q, r]:
						current_is_articulation = True
					if status[q1, r1] == VISITING:
						low[q, r] = np.min(np.array([low[q, r], low[q1, r1]], dtype=np.int8))
					if np.all(s != parent[q, r]):
						low[q, r] = np.min(np.array([low[q, r], depth[q1, r1]], dtype=np.int8))
				if current_is_root and root_child_count > 1:
					cutpoints[q, r] = True
				elif not current_is_root and current_is_articulation:
					cutpoints[q, r] = True
		return cutpoints

	def _print_nums(self,data):
		print(f'-'*11)
		for r in range(BOARD_SIZE):
			l = ''
			for q in range(BOARD_SIZE):
				l += f'{data[q, r]} '
			print(r * ' ' + l)

	def _queen_in_hand(self, player):
		queen = self._get_player_pieces(player)[0]
		return self._piece_in_hand(queen)
