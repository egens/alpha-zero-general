import numba
import numpy as np

from .HiveConstants import *
from .HiveConstants import _decode_action, _encode_action, _np_all_axis1, _is_queen, _is_beetle, _is_ant, \
	_is_grasshopper, _is_spider
from .HiveDisplay import _print_flag, _print_flags, _print_moves


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
# 0,0 means player hand

# ACTION is bitfield — last is pass
# 	field B: which bug used (PLAYER_PIECES_COUNT)
#	field Q: Q axial coord (BOARD_SIZE)
#	field R: R axial coord (BOARD_SIZE)
#	W Q R
#
# New State
# TODO Height of bugs and mosquitoes
# 2*PLAYER_PIECES_COUNT*2
# For each piece a pair of
# 	1. Parent piece in BFS tree from the first game piece
#	2. One of 6 directions — position from parent Piece
# First game piece has itself as parent
# Instead of direction first game piece second element is round num
#
# Directions
#   0 \ / 1
#  5 - P - 2
#   4 / \ 3

@njit(cache=True, fastmath=True, nogil=True)
def observation_size():
	return (2*PLAYER_PIECES_COUNT + 1,2)

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return MOVES_COUNT

spec = [
	('state'      , numba.int8[:,:]),
	('pieces'     , numba.bool_[:,:]),
	('round_num'  , numba.int8[:]),
	('beetle_height'  , numba.int8[:]),
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
		self.round_num = np.zeros(2, dtype=np.int8)
		self.beetle_height = np.zeros(4, dtype=np.int8)

	def get_state(self):
		return self.state

	def valid_moves(self, player):
		actions = np.zeros(action_size(), dtype=np.bool_)
		player_pieces = self._get_player_pieces(player)
		# First move white to the middle
		if self.get_round() == 0:
			for piece in [GRASSHOPPER_1, SPIDER_1]: # Prohibit first ant or queen or beetle
				actions[_encode_action(piece, piece, 0)] = True
			return actions
		# Second move black to the right of white
		elif self.get_round() == 1:
			for piece in [GRASSHOPPER_1, SPIDER_1]: # Prohibit first ant or queen or beetle
				to_piece, direction = self._get_first_adj_piece(player, piece, BOARD_SIZE//2+1, BOARD_SIZE//2)
				actions[_encode_action(piece, to_piece, direction)] = True
			return actions

		# On the 4th turn each player must put Q if its in hand
		if self.get_round() == 6 and self._piece_in_hand(player_pieces[0]):
			player_pieces = np.array([player_pieces[0]])
		if self.get_round() == 7 and self._piece_in_hand(player_pieces[0]):
			player_pieces = np.array([player_pieces[0]])

		cutpoints = self._get_cutpoints()
		a_spawn_shown = False
		b_spawn_shown = False
		g_spawn_shown = False
		s_spawn_shown = False
		for piece in player_pieces:
			moves = [self.state[0]]
			moves.pop()
			if self._piece_in_hand(piece):
				# Pieces should spawn in their order
				if _is_ant(piece):
					if a_spawn_shown:
						continue
					a_spawn_shown = True
				if _is_beetle(piece):
					if b_spawn_shown:
						continue
					b_spawn_shown = True
				if _is_grasshopper(piece):
					if g_spawn_shown:
						continue
					g_spawn_shown = True
				if _is_spider(piece):
					if s_spawn_shown:
						continue
					s_spawn_shown = True
				moves += self._get_spawns(player)

			else: # Piece in play
				if cutpoints[self.state[piece][0], self.state[piece][1]]:
					continue
				# if self._beetle_above(piece):
				# 	continue
				if self._queen_in_hand(player):
					continue
				if _is_queen(piece) or _is_beetle(piece):
					surr = self._get_surroundings(self.state[piece][0], self.state[piece][1])
					for i in range(6):
						if self._is_piece(surr[i][0], surr[i][1]):
							continue
						nexti = (i + 1) % 6
						previ = (i - 1) % 6
						if self._is_piece(surr[previ][0], surr[previ][1]) and not self._is_piece(surr[nexti][0], surr[nexti][1]):
							moves += [np.array([surr[i][0], surr[i][1]], dtype=np.int8)]
						if not self._is_piece(surr[previ][0], surr[previ][1]) and self._is_piece(surr[nexti][0], surr[nexti][1]):
							moves += [np.array([surr[i][0], surr[i][1]], dtype=np.int8)]
				# # Move beetles up
				# if _is_beetle(piece):
				# 	for s in self._get_surrounding_pieces(self.state[piece][0], self.state[piece][1]):
				# 		q, r = s
				# 		moves += [np.array([q, r], dtype=np.int8)]
				elif _is_ant(piece):
					moves += self._get_ant_moves(piece)
				elif _is_grasshopper(piece):
					coord = self.state[piece]
					for d in DIRECTIONS:
						if self._is_piece(coord[0] + d[0], coord[1] + d[1]):
							for i in range(1, BOARD_SIZE):
								if not self._is_piece(coord[0] + i*d[0], coord[1] + i*d[1]):
									moves += [np.array([coord[0] + i*d[0], coord[1] + i*d[1]], dtype=np.int8)]
									break
				elif _is_spider(piece):
					# TODO Remove backtracking in holes
					moves += self._get_ant_moves(piece, steps=3)
			for q, r in moves:
				if self._is_board(q, r):
					to_piece, direction = self._get_first_adj_piece(player, piece, q, r)
					# if piece == 3:
					# 	print(to_piece, direction)
					actions[_encode_action(piece, to_piece, direction)] = True

		if sum(actions) == 0:
			actions[PASS_ACTION] = True
		return actions

	def _get_spawns(self, player):
		visited = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		enemies = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		flag = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		for enemy_piece in self._get_player_pieces(self._get_opponent(player)):
			if self._piece_in_hand(enemy_piece):
				continue
			q, r = self.state[enemy_piece]
			enemies[q, r] = True
		player_perimeter = [self.state[0]]
		player_perimeter.pop()
		for piece in self._get_player_pieces(player):
			if self._piece_in_hand(piece):
				continue
			q, r = self.state[piece]
			for s in self._get_surroundings(q, r):
				q1, r1 = s
				if not self._is_board(q1, r1) or self._is_piece(q1, r1) or visited[q1, r1]:
					continue
				player_perimeter += [s]
				visited[q1, r1] = True

		spawns = [self.state[0]]
		spawns.pop()
		for potential_spawn in player_perimeter:
			q, r = potential_spawn
			no_enemies = True
			for s in self._get_surroundings(q, r):
				q1, r1 = s
				if self._is_board(q1, r1) and enemies[q1, r1]:
					no_enemies = False
					break
			if no_enemies:
				spawns += [potential_spawn]
				flag[q, r] = True
		# _print_flag(flag, c='s')
		return spawns

	def _get_ant_moves(self, piece, steps=-1):
		visited = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		depth = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.int8)
		moves = [self.state[0]]
		moves.pop()
		stack = [self.state[piece]]
		q, r = self.state[piece]
		self.pieces[q, r] = False
		while len(stack) > 0:
			q, r = stack.pop()
			visited[q, r] = True
			surr = self._get_surroundings(q, r)
			for i in range(6):
				q1, r1 = surr[i][0], surr[i][1]
				if not self._is_board(q1, r1):
					continue
				depth[q1, r1] = depth[q, r] + 1
				if self._is_piece(q1, r1) or visited[q1, r1]:
					continue
				nexti = (i + 1) % 6
				previ = (i - 1) % 6
				if ((self._is_piece(surr[previ][0], surr[previ][1]) and not self._is_piece(surr[nexti][0], surr[nexti][1])) or
						(not self._is_piece(surr[previ][0], surr[previ][1]) and self._is_piece(surr[nexti][0], surr[nexti][1]))):
					if steps == -1:
						stack += [surr[i]]
						moves += [surr[i]]
					else:
						if depth[q1, r1] == steps:
							moves += [surr[i]]
						else:
							stack += [surr[i]]
		q, r = self.state[piece]
		self.pieces[q, r] = True
		return moves

	def make_move(self, move, player, random_seed):
		# _print_flags(self)
		# _print_moves(self, player)
		self._increment_round()
		if move == PASS_ACTION:
			return self._get_opponent(player)
		piece, to_piece, direction, is_opponent_piece = _decode_action(move)
		# print(piece, to_piece, direction, is_opponent_piece)
		if player == 1:
			piece += PLAYER_PIECES_COUNT
		if is_opponent_piece and player == 0:
			to_piece = to_piece + PLAYER_PIECES_COUNT
		if not is_opponent_piece and player == 1:
			to_piece = to_piece + PLAYER_PIECES_COUNT
		q, r = self.state[piece]
		self.pieces[q, r] = False
		new_q, new_r = self.state[to_piece] + DIRECTIONS[direction]
		# First move override
		if to_piece == piece:
			new_q, new_r = BOARD_SIZE//2, BOARD_SIZE//2
		self.state[piece] = [new_q, new_r]
		# self._check_state()
		self.pieces[new_q, new_r] = True
		# if self.pieces[new_q, new_r] == True:
		# 	pass
		# 	# TODO going above the hive
		return self._get_opponent(player)


	def check_end_game(self, next_player):
		# Ideally game should be over earlier
		if self.get_round() > 1000:
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
		# print(policy[np.where(policy > 0)])
		# TODO Implement Hive

		return symmetries

	def get_round(self):
		return np.int16(self.round_num[1]) * 128 + self.round_num[0]

	def _increment_round(self):
		if self.round_num[0] == 127:
			self.round_num[0] = 0
			self.round_num[1] += 1
		else:
			self.round_num[0] += 1

	def copy_state(self, state, copy_or_not):
		if self.state is state and not copy_or_not:
			return
		self.state = state.copy() if copy_or_not else state

		self.pieces = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		for i, qr in enumerate(self.state[:-1]):
			if self._piece_in_play(i):
				self.pieces[qr[0], qr[1]] = True
		self.round_num = self.state[-1,:]
	# self.beetle_height = [self.state[-3,0], self.state[-3,1], self.state[-2,0], self.state[-2,1]]

	def _is_piece(self, q, r):
		return self._is_board(q, r) and self.pieces[q, r]

	def _is_board(self, q, r):
		return 0 <= q < BOARD_SIZE and 0 <= r < BOARD_SIZE

	def _get_bug_position(self, searched_bug):
		for i in np.ndindex(BOARD_SIZE, BOARD_SIZE):
			if self.state[i] == searched_bug:
				return i
		print(f'Should not happen gwp {searched_bug}')
		return (-1, -1)

	# Ordered clockwise
	def _get_surroundings(self, q, r):
		return np.array([[q, r-1], [q+1, r-1],
						 [q+1, r], [q, r+1],
						 [q-1, r+1], [q-1, r]], dtype=np.int8)
	# result = result[_np_all_axis1(result >= 0)]
	# result = result[_np_all_axis1(result < BOARD_SIZE)]
	# return result

	def _get_surrounding_pieces(self, q, r):
		idxs = np.zeros(6, dtype=np.int8)
		surrs = self._get_surroundings(q,r)
		count = 0
		for i, coord in enumerate(surrs):
			q1, r1 = coord
			if self._is_board(q1, r1) and self.pieces[q1, r1]:
				idxs[count] = i
				count += 1
		return surrs[idxs[:count]]

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

		if len(stack) == 0:
			return cutpoints
		q, r = stack[-1]
		depth[q, r] = 0
		low[q, r] = 0
		while len(stack) > 0:
			q, r = stack[-1]
			pq, pr = parent[q, r]
			current_is_root = pr == -1 and pr == -1
			surr_visited = True
			for s in self._get_surrounding_pieces(q, r):
				q1, r1 = s
				if status[q1, r1] == UNVISITED:
					status[q1, r1] = VISITING
					stack += [s]
					surr_visited = False
					parent[q1, r1] = [q, r]
					depth[q1, r1] = depth[q, r] + 1
					if current_is_root:
						root_child_count += 1
					break # Enabling DFS

			# The lowpoint of v can be computed after visiting all descendants of v
			if surr_visited:
				q, r = stack.pop()
				status[q, r] = VISITED
				# as the minimum of the depth of v,
				low[q, r] = depth[q, r]
				current_is_articulation = False
				for s in self._get_surrounding_pieces(q, r):
					q1, r1 = s
					if np.all(s == parent[q, r]): # parent of current
						pass
					# low[q1, r1] = np.min(np.array([low[q, r], low[q1, r1]], dtype=np.int8))
					elif q == parent[q1, r1][0] and r == parent[q1, r1][1]: # child of current
						# and the lowpoint of all children of v in the depth-first-search tree.
						low[q, r] = np.min(np.array([low[q, r], low[q1, r1]], dtype=np.int8))
						# v is a cut vertex if and only if there is a child y of v such that lowpoint(y) ≥ depth(v)
						if low[q1, r1] >= depth[q, r]:
							current_is_articulation = True
					else: # other
						# the depth of all neighbors of v (other than the parent of v in the depth-first-search tree)
						low[q, r] = np.min(np.array([low[q, r], depth[q1, r1]], dtype=np.int8))
				if current_is_root and root_child_count > 1:
					cutpoints[q, r] = True
				elif not current_is_root and current_is_articulation:
					cutpoints[q, r] = True
		# self._print_nums(depth, other=low)
		return cutpoints

	def _print_nums(self,data, other=None):
		print(f'-'*11)
		for r in range(BOARD_SIZE):
			l = ''
			for q in range(BOARD_SIZE):
				l += f'{data[q, r] if data[q, r] >= 0 else ' '} '
				if other is not None:
					l += f'{'(' + str(other[q, r]) + ')' if other[q, r] >= 0 else '   '} '
			print(r * ' ' + l)

	def _queen_in_hand(self, player):
		queen = self._get_player_pieces(player)[0]
		return self._piece_in_hand(queen)

	def _near_opponent_queen(self, player, q, r):
		queen = self._get_player_pieces(self._get_opponent(player))[0]
		q1, r1 = self.state[queen]
		if q1 + r1 == 0:
			return False
		return abs(q - q1) + abs(r - r1) == 1

	def _closer_to_opponent_queen(self, player, piece, new_q, new_r):
		queen = self._get_player_pieces(self._get_opponent(player))[0]
		q1, r1 = self.state[queen]
		if q1 + r1 == 0:
			return False
		q, r = self.state[piece]
		return self.__distance(new_q, new_r, q1, r1) < self.__distance(q, r, q1, r1)

	def __distance(self, q1, r1, q2, r2):
		dq = abs(q1 - q2)
		dr = abs(r1 - r2)
		s1 = - q1 - r1
		s2 = - q2 - r2
		ds = abs(s1 - s2)
		return (dq + dr + ds) / 2

	def _get_first_adj_piece(self, player, piece, q, r):
		for i, d in enumerate(DIRECTIONS):
			q1, r1 = q + d[0], r + d[1]
			for p, s in enumerate(self.state):
				if p == piece + player * PLAYER_PIECES_COUNT:
					continue
				if q1 == s[0] and r1 == s[1]:
					return p, (i + 3) % 6
		raise Exception(f'{q,r} should have adjacent piece')

	def _check_state(self):
		for i, s1 in enumerate(self.state[:-1]):
			for j, s2 in enumerate(self.state[:-1]):
				if i == j: continue
				if np.all(s1 == s2) and sum(s1) + sum(s2) != 0:
					print(self.state, s1, s2)
					raise Exception('WRONG STATE')
