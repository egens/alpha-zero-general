import numpy as np
from colorama import Style, Fore, Back
from .HiveConstants import *
from .HiveConstants import _decode_action


def move_to_str(move, player):
	piece, new_q, new_r = _decode_action(move)
	piece = player * PLAYER_PIECES_COUNT + piece
	return f'move {piece} to {new_q} {new_r}'


############################# PRINT GAME ######################################

def _print_main(board):
	print(f'-'*11)
	for r in range(BOARD_SIZE):
		l = ''
		for q in range(BOARD_SIZE):
			if q < (BOARD_SIZE/2 - r - 1) or q > (3/2*BOARD_SIZE - r - 1):
				c = f'{Fore.RED}. {Style.RESET_ALL}'
			else:
				c = '. '
			fg = Fore.BLACK
			bg = Back.WHITE
			for n, piece in enumerate(board.state[:-1]):
				if n == PLAYER_PIECES_COUNT:
					fg = Fore.WHITE
					bg = Back.BLACK
				if piece[0] == q and piece[1] == r:
					c = f'{fg}{bg}{_get_char(n)}{Style.RESET_ALL}'
			l += f'{c}'
		# if x % 2 == 1:
		# 	l = ' ' + l
		print(r * ' ' + l)
	# input()

def _get_char(piece):
	n = piece % PLAYER_PIECES_COUNT
	if n == QUEEN:
		return 'Q '
	if n in ANTS:
		return 'A'+ str(n + 1)

def print_board(board):
	print()
	_print_main(board)