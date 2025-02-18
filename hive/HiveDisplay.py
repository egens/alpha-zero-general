import numpy as np
from colorama import Style, Fore, Back
from .HiveConstants import *
from .HiveConstants import _decode_action, _is_queen, _is_beetle, _is_ant, _is_grasshopper, _is_spider


def move_to_str(move, player):
	piece, new_q, new_r = _decode_action(move)
	piece = player * PLAYER_PIECES_COUNT + piece
	return f'move {_get_char(piece)} to {new_q} {new_r}'

############################# PRINT GAME ######################################

def _print_main(board):
	print(f'-'*11)
	print()
	for r in range(BOARD_SIZE):
		l = ''
		for q in range(BOARD_SIZE):
			if r + q == 0:
				l += f'{Fore.BLACK}. {Style.RESET_ALL}'
				continue
			if q < (BOARD_SIZE/2 - r - 1) or q > (3/2*BOARD_SIZE - r - 1):
				c = f'{Fore.BLACK}. {Style.RESET_ALL}'
			else:
				c = f'{Fore.BLUE}. {Style.RESET_ALL}'
			for piece, coord in enumerate(board.state[:-1]):
				if coord[0] == q and coord[1] == r:
					c = _get_char(piece)
			l += f'{c}'
		print(r * ' ' + l)
	# input()

def _get_char(piece):
	fg = Fore.BLACK
	bg = Back.WHITE
	if piece >= PLAYER_PIECES_COUNT:
		fg = Fore.WHITE
		bg = Back.BLACK
	c = ''
	if _is_queen(piece):
		c = 'Q '
	elif _is_ant(piece):
		c = 'A'+ str(piece%PLAYER_PIECES_COUNT + 1)
	elif _is_grasshopper(piece):
		c = 'G'+ str(piece%PLAYER_PIECES_COUNT - 3)
	elif _is_beetle(piece):
		c = 'B'+ str(piece%PLAYER_PIECES_COUNT - 6)
	elif _is_spider(piece):
		c = 'S'+ str(piece%PLAYER_PIECES_COUNT - 8)
	return f'{fg}{bg}{c}{Style.RESET_ALL}'

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
	print(f'-'*11)
	for r in range(BOARD_SIZE):
		l = ''
		for q in range(BOARD_SIZE):
			if board.pieces[q, r]:
				if board.cutpoints[q, r]:
					l += f'C '
				else:
					l += f'* '
			else:
				l += f'. '
		print(r * ' ' + l)

def _print_moves(board):
	lines = [_get_char(p) + '' for p in range(2*PLAYER_PIECES_COUNT)]
	for p in range(2):
		valid = board.valid_moves(p)
		for a, valid in enumerate(valid):
			if valid:
				piece, new_q, new_r = _decode_action(a)
				lines[PLAYER_PIECES_COUNT*p + piece] += f' ({new_q},{new_r})'
	for l in lines:
		if len(l) > 2:
			print(l)

def _print_hands(board):
	l = 'Player hands: '
	for piece, coord in enumerate(board.state[:-1]):
		if coord[0] + coord[1] == 0:
			l += f'{_get_char(piece)} '
	print(l)

def print_board(board):
	print()
	# _print_flags(board)
	_print_moves(board)
	_print_hands(board)
	_print_main(board)