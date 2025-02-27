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
	return (2*PLAYER_PIECES_COUNT + 1 + 2,2)

@njit(cache=True, fastmath=True, nogil=True)
def action_size():
	return MOVES_COUNT

# spec = [
# 	('state'      , numba.int8[:,:]),
# 	('pieces'     , numba.int8[:,:]),
# 	('positions'  , numba.int8[:,:]),
# 	('round_num'  , numba.int8[:]),
# 	('beetle_height'  , numba.int8[:,:]),
# ]
# @numba.experimental.jitclass(spec)
class Board():
	def __init__(self, num_players):
		self.state = np.zeros((2*PLAYER_PIECES_COUNT + 1 + 2,2), dtype=np.int8) # QR coordinates
		self.init_game()

	def get_score(self, player):
		# TODO Number of bugs around queen
		# self._get_bug_position(-1 * player * QUEEN)
		return 0

	def init_game(self):
		self.copy_state(np.zeros((2*PLAYER_PIECES_COUNT + 1 + 2,2), dtype=np.int8), copy_or_not=False)

		self.beetle_height[:,:] = -1
		self.pieces = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.int8)
		self.pieces[:,:] = -1

	def get_state(self):
		return self.state

	def valid_moves(self, player):
		actions = np.zeros(action_size(), dtype=np.bool_)
		player_pieces = self._get_player_pieces(player)
		# First move white to the middle
		if self.get_round() == 0:
			for piece in player_pieces:
				if piece % PLAYER_PIECES_COUNT not in [GRASSHOPPER_1, SPIDER_1]: # Prohibit first ant or queen or beetle
					continue
				actions[_encode_action(piece, piece, 0)] = True
			return actions
		# Second move black to the right of white
		elif self.get_round() == 1:
			for piece in player_pieces:
				if piece % PLAYER_PIECES_COUNT not in [GRASSHOPPER_1, SPIDER_1]: # Prohibit first ant or queen or beetle
					continue
				to_piece, direction = self._get_first_adj_piece(player, piece, BOARD_SIZE//2 + 1, BOARD_SIZE//2)
				actions[_encode_action(piece, to_piece, direction)] = True
			return actions

		# Queen on the 4th move
		player_pieces_in_play = len(self._pieces_in_play(player))
		if player_pieces_in_play == 3 and self._piece_in_hand(player_pieces[0]):
			player_pieces = np.array([player_pieces[0]])

		cutpoints = self._get_cutpoints()
		spawns = self._get_spawns(player)
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
				moves += spawns

			else: # Piece in play
				if cutpoints[self.positions[piece][0], self.positions[piece][1]]:
					continue
				if self._beetle_above(piece):
					continue
				if self._queen_in_hand(player):
					continue

				if _is_queen(piece):
					for i in range(6):
						q, r = self.positions[piece] + DIRECTIONS[i]
						if self._is_piece(q, r):
							continue
						q_next, r_next = self.positions[piece] + DIRECTIONS[(i + 1) % 6]
						q_prev, r_prev = self.positions[piece] + DIRECTIONS[(i - 1) % 6]
						if self._is_piece(q_prev, r_prev) and not self._is_piece(q_next, r_next):
							moves += [np.array([q, r], dtype=np.int8)]
						if not self._is_piece(q_prev, r_prev) and self._is_piece(q_next, r_next):
							moves += [np.array([q, r], dtype=np.int8)]
				elif _is_beetle(piece):
					for i in range(6):
						q, r = self.positions[piece] + DIRECTIONS[i]
						if self._is_piece(q, r):
							continue
						q_next, r_next = self.positions[piece] + DIRECTIONS[(i + 1) % 6]
						q_prev, r_prev = self.positions[piece] + DIRECTIONS[(i - 1) % 6]
						if self._is_piece(q_prev, r_prev) and not self._is_piece(q_next, r_next):
							moves += [np.array([q, r], dtype=np.int8)]
						if not self._is_piece(q_prev, r_prev) and self._is_piece(q_next, r_next):
							moves += [np.array([q, r], dtype=np.int8)]
					for d in DIRECTIONS:
						q, r = self.positions[piece] + d
						if self._is_piece(q, r):
							moves += [np.array([q, r], dtype=np.int8)]
				elif _is_ant(piece):
					moves += self._get_ant_moves(piece)
				elif _is_grasshopper(piece):
					coord = self.positions[piece]
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
					actions[_encode_action(piece, to_piece, direction)] = True

		if sum(actions) == 0:
			actions[PASS_ACTION] = True
		return actions

	def _get_spawns(self, player):
		visited = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		enemies = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		for enemy_piece in self._get_player_pieces(self._get_opponent(player)):
			if self._piece_in_hand(enemy_piece):
				continue
			q, r = self.positions[enemy_piece]
			enemies[q, r] = True
		player_perimeter = [self.positions[0]]
		player_perimeter.pop()
		for piece in self._get_player_pieces(player):
			if self._piece_in_hand(piece):
				continue
			for d in DIRECTIONS:
				q1, r1 = self.positions[piece] + d
				if not self._is_board(q1, r1) or self._is_piece(q1, r1) or visited[q1, r1]:
					continue
				player_perimeter += [np.array([q1, r1], dtype=np.int8)]
				visited[q1, r1] = True

		spawns = [self.positions[0]]
		spawns.pop()
		for potential_spawn in player_perimeter:
			no_enemies = True
			for d in DIRECTIONS:
				q1, r1 = potential_spawn + d
				if self._is_board(q1, r1) and enemies[q1, r1]:
					no_enemies = False
					break
			if no_enemies:
				spawns += [potential_spawn]
		return spawns

	def _get_ant_moves(self, piece, steps=-1):
		visited = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
		depth = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.int8)
		moves = [self.positions[0]]
		moves.pop()
		stack = [self.positions[piece]]
		q, r = self.positions[piece]
		self.pieces[q, r] = -1
		while len(stack) > 0:
			s = stack.pop()
			q, r = s
			visited[q, r] = True
			for i, d in enumerate(DIRECTIONS):
				q1, r1 = s + d
				if not self._is_board(q1, r1):
					continue
				depth[q1, r1] = depth[q, r] + 1
				if self._is_piece(q1, r1) or visited[q1, r1]:
					continue
				q_next, r_next = s + DIRECTIONS[(i + 1) % 6]
				q_prev, r_prev = s + DIRECTIONS[(i - 1) % 6]
				if ((self._is_piece(q_prev, r_prev) and not self._is_piece(q_next, r_next)) or
						(not self._is_piece(q_prev, r_prev) and self._is_piece(q_next, r_next))):
					if steps == -1:
						stack += [s + d]
						moves += [s + d]
					else:
						if depth[q1, r1] == steps:
							moves += [s + d]
						else:
							stack += [s + d]
		q, r = self.positions[piece]
		self.pieces[q, r] = piece
		return moves

	def make_move(self, move, player, random_seed):
		self._increment_round()

		if move == PASS_ACTION:
			return self._get_opponent(player)

		piece, to_piece, direction, is_opponent_piece = _decode_action(move, player)
		q, r = self.positions[piece]
		self.pieces[q, r] = -1
		new_q, new_r = self.positions[to_piece] + DIRECTIONS[direction]
		# First move override
		if to_piece == piece:
			new_q, new_r = BOARD_SIZE//2, BOARD_SIZE//2
		if self._is_piece(new_q, new_r) and not _is_beetle(piece):
			raise Exception('New position should be empty if the piece is not beetle')
		if _is_beetle(piece):
			beetle_number = piece % PLAYER_PIECES_COUNT % BEETLE_1
			piece_below_from = self.state[2*PLAYER_PIECES_COUNT + player, beetle_number]
			if piece_below_from >= 0:
				self.pieces[q, r] = piece_below_from
			if self.pieces[new_q, new_r] >= 0:
				self.state[2*PLAYER_PIECES_COUNT + player, beetle_number] = self.pieces[new_q, new_r]
			else: # Nothing below to
				self.state[2*PLAYER_PIECES_COUNT + player, beetle_number] = -1

		self.positions[piece] = [new_q, new_r]
		self.pieces[new_q, new_r] = piece
		return self._get_opponent(player)

	def check_end_game(self, next_player):
		# Ideally game should be over earlier
		if self.get_round() > 100:
			return np.array([0.1, 0.1], dtype=np.float32)
		cur_player_queen = self.positions[self._get_player_pieces(self._get_opponent(next_player))[0]]
		cur_player_health = 6 - len(self._get_surrounding_pieces(cur_player_queen[0], cur_player_queen[1]))
		next_player_queen = self.positions[self._get_player_pieces(next_player)[0]]
		next_player_health = 6 - len(self._get_surrounding_pieces(next_player_queen[0], next_player_queen[1]))
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
		new_state[:PLAYER_PIECES_COUNT] = self.state[PLAYER_PIECES_COUNT:2*PLAYER_PIECES_COUNT]
		new_state[PLAYER_PIECES_COUNT:2*PLAYER_PIECES_COUNT] = self.state[:PLAYER_PIECES_COUNT]
		# Swap beetles below pieces
		new_state[2*PLAYER_PIECES_COUNT,:] = self.state[2*PLAYER_PIECES_COUNT + 1,:]
		new_state[2*PLAYER_PIECES_COUNT + 1,:] = self.state[2*PLAYER_PIECES_COUNT,:]
		beetles = new_state[2*PLAYER_PIECES_COUNT:2*PLAYER_PIECES_COUNT + 2,:]
		for (i,j), value in np.ndenumerate(beetles):
			if value >= 0:
				beetles[i,j] = (value + PLAYER_PIECES_COUNT) % (2*PLAYER_PIECES_COUNT)
		self.state = new_state
		self.positions = self.state[:2*PLAYER_PIECES_COUNT,:]
		self.round_num = self.state[-1,:]
		self.beetle_height = self.state[2*PLAYER_PIECES_COUNT:2*PLAYER_PIECES_COUNT+2,:]
		self._reload_pieces()

	def _reload_pieces(self):
		self.pieces = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.int8)
		self.pieces[:,:] = -1
		for piece, qr in enumerate(self.positions):
			if self._piece_in_play(piece) and not _is_beetle(piece):
				self.pieces[qr[0], qr[1]] = piece
		# Handle beetles
		visited = np.array([False, False, False, False], dtype=np.bool_)
		while not np.all(visited):
			for i, beetle in enumerate(BEETLES):
				if visited[i]:
					continue
				if self._piece_in_hand(beetle):
					visited[i] = True
					continue
				q, r = self.positions[beetle]
				beetle_number = beetle % PLAYER_PIECES_COUNT % BEETLE_1
				player = beetle // PLAYER_PIECES_COUNT
				piece_below = self.beetle_height[player, beetle_number]
				if self.pieces[q,r] == piece_below:
					self.pieces[q,r] = beetle
					visited[i] = True

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

		self.positions = self.state[:2*PLAYER_PIECES_COUNT,:]
		self.round_num = self.state[-1,:]
		self.beetle_height = self.state[2*PLAYER_PIECES_COUNT:2*PLAYER_PIECES_COUNT+2,:]
		self._reload_pieces()

	def _is_piece(self, q, r):
		return self._is_board(q, r) and self.pieces[q, r] >= 0

	def _is_board(self, q, r):
		return 0 <= q < BOARD_SIZE and 0 <= r < BOARD_SIZE

	def _get_bug_position(self, searched_bug):
		for i in np.ndindex(BOARD_SIZE, BOARD_SIZE):
			if self.positions[i] == searched_bug:
				return i
		print(f'Should not happen gwp {searched_bug}')
		return (-1, -1)

	def _get_surrounding_pieces(self, q, r):
		result = [self.positions[0]]
		result.pop()
		for i, d in enumerate(DIRECTIONS):
			q1, r1 = q + d[0], r + d[1]
			if self._is_board(q1, r1) and self.pieces[q1, r1] >= 0:
				result += [np.array([q1, r1], dtype=np.int8)]
		return result

	def _get_player_pieces(self, player):
		return (player * PLAYER_PIECES_COUNT + np.arange(PLAYER_PIECES_COUNT, dtype=np.int8)).astype(np.int8)

	# players are 0 and 1
	def _get_opponent(self, player):
		return 1 - player

	def _piece_in_play(self, piece):
		return not self._piece_in_hand(piece)

	def _pieces_in_play(self, player=None):
		pieces = np.arange(2*PLAYER_PIECES_COUNT, dtype=np.int8)
		in_play = [pieces[0]]
		in_play.pop()
		if player is not None:
			pieces = self._get_player_pieces(player)
		for piece in pieces:
			if self._piece_in_play(piece):
				in_play += [piece]
		return in_play

	def _piece_in_hand(self, piece):
		return self.positions[piece][0] == 0 and self.positions[piece][1] == 0

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
		stack = [self.positions[0]]
		stack.pop()
		for p in range(2*PLAYER_PIECES_COUNT):
			if self._piece_in_play(p):
				stack = [self.positions[p]]
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
		q1, r1 = self.positions[queen]
		if q1 + r1 == 0:
			return False
		return abs(q - q1) + abs(r - r1) == 1

	def _closer_to_opponent_queen(self, player, piece, new_q, new_r):
		queen = self._get_player_pieces(self._get_opponent(player))[0]
		q1, r1 = self.positions[queen]
		if q1 + r1 == 0:
			return False
		q, r = self.positions[piece]
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
			if not self._is_board(q1, r1) or not self._is_piece(q1, r1):
				continue
			p = self.pieces[q1, r1]
			if p == piece + player * PLAYER_PIECES_COUNT:
				continue
			return p, (i + 3) % 6
		raise Exception(f'{q,r} should have adjacent piece')

	def _check_state(self):
		for i, s1 in enumerate(self.positions):
			for j, s2 in enumerate(self.positions):
				if i == j: continue
				if np.all(s1 == s2) and sum(s1) + sum(s2) != 0:
					print(self.state, s1, s2)
					raise Exception('WRONG STATE')

	def _beetle_above(self, piece):
		q, r = self.positions[piece]
		if self.pieces[q,r] != piece:
			if _is_beetle(self.pieces[q,r]):
				return True
			else:
				Exception('Only beetle can be above')
		return False
