import numpy as np
from colorama import Style, Fore, Back
from .HiveConstants import *
from .HiveConstants import _decode_action, _is_queen, _is_beetle, _is_ant, _is_grasshopper, _is_spider


def move_to_str(move, player):
	piece, to_piece, direction, is_opponent_piece = _decode_action(move, player)
	# if to_piece == piece:
	# 	return f'place {_get_char(piece)}'
	piece = player * PLAYER_PIECES_COUNT + piece
	return f'move {_get_char(piece)} to {_get_move(move, player)}'

############################# PRINT GAME ######################################


def _print_main(board):
	l = ''
	for i in range(BOARD_SIZE):
		l += str(i) + ' '
	print(l)
	flag_lines = __get_flags_lines(board)
	for i, r in enumerate(range(BOARD_SIZE)):
		l = ''
		for q in range(BOARD_SIZE):
			if r + q == 0:
				l += f'{Fore.BLACK}. {Style.RESET_ALL}'
				continue
			if q < (BOARD_SIZE/2 - r - 1) or q > (3/2*BOARD_SIZE - r - 1):
				c = f'{Fore.BLACK}. {Style.RESET_ALL}'
			else:
				c = f'{Fore.BLUE}. {Style.RESET_ALL}'
			for piece, coord in enumerate(board.positions):
				if coord[0] == q and coord[1] == r:
					c = _get_char(piece)
			l += f'{c}'
		left_padding = r * ' '
		print(left_padding + l + ' ' + flag_lines[i])
	# input()

def _get_char(piece):
	fg = Fore.BLACK
	bg = Back.WHITE
	if piece >= PLAYER_PIECES_COUNT:
		fg = Fore.WHITE
		bg = Back.BLACK
	c = ''
	if _is_queen(piece):
		c = 'QQ'
	elif _is_ant(piece):
		c = 'A'+ str(piece%PLAYER_PIECES_COUNT)
	elif _is_grasshopper(piece):
		c = 'G'+ str(piece%PLAYER_PIECES_COUNT - 3)
	elif _is_beetle(piece):
		c = 'B'+ str(piece%PLAYER_PIECES_COUNT - 6)
	elif _is_spider(piece):
		c = 'S'+ str(piece%PLAYER_PIECES_COUNT - 8)
	return f'{fg}{bg}{c}{Style.RESET_ALL}'

def _get_move(move, player):
	piece, to_piece, direction, is_opponent_piece = _decode_action(move, player)
	if direction < 3:
		return f'{_get_char(to_piece)}{DIRECTIONS_STRING[direction]}'
	else:
		return f'{DIRECTIONS_STRING[direction]}{_get_char(to_piece)}'

def _print_flag(flag, c='f'):
	print(f'-'*11)
	for r in range(BOARD_SIZE):
		l = ''
		for q in range(BOARD_SIZE):
			if flag[q, r]:
				l += f'{c} '
			else:
				l += f'. '
		print(r * ' ' + l)

def _print_flags(board):
	for l in __get_flags_lines(board):
		print(l)

def __get_flags_lines(board):
	spawns = np.zeros((BOARD_SIZE,BOARD_SIZE), dtype=np.bool_)
	lines = []
	for s in board._get_spawns(board.get_round() % 2):
		spawns[s[0], s[1]] = True
	for r in range(BOARD_SIZE):
		l = ''
		for q in range(BOARD_SIZE):
			if q < (BOARD_SIZE/2 - r - 1) or q > (3/2*BOARD_SIZE - r - 1):
				l += '  '
				continue
			if board.pieces[q, r] >= 0:
				if board._get_cutpoints()[q, r]:
					l += f'C '
				else:
					l += f'* '
			elif spawns[q, r]:
				l += f's '
			else:
				l += f'. '
		lines += [l]
	return lines

def _print_moves(board, actions=None):
	player = board.get_round() % 2
	player_pieces = board._get_player_pieces(player)
	lines = [f'{p} {_get_char(p)}: ' for p in player_pieces]
	actions = board.valid_moves(player)
	for a, valid in enumerate(actions):
		if valid:
			if a == PASS_ACTION:
				print('PASS')
				return
			piece, to_piece, direction, is_opponent_piece = _decode_action(a)
			lines[piece] += f' {_get_move(a, player)}'
	for l in lines:
		# if l[-1] != ' ':
		print(l)

def _print_hands(board):
	l = 'Player hands: '
	for piece, coord in enumerate(board.positions):
		if coord[0] + coord[1] == 0:
			l += f'{_get_char(piece)} '
	print(l)

def print_board(board):
	print()
	_print_main(board)
	_print_hands(board)
	_print_moves(board)