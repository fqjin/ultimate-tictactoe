"""Code to run UTTT games"""
import numpy as np


def board2bit(board):
    """Custom hashing algorithm to convert boards to 18-bit integers"""
    return sum(board[i] << 2*i for i in range(9))


decode_dict = {
    0: ' ',
    1: 'X',
    2: 'O',
    3: 'T',
}


def make_bit2board_table():
    """Returns dictionary linking integers (bits) with 3x3 boards"""
    table = {}
    for i, indices in enumerate(np.ndindex(*[4] * 9)):
        table[i] = indices[::-1]
    return table


bit2board = make_bit2board_table()


def draw_board(board):
    chars = [decode_dict[v] for v in board]
    print("""
     {} │ {} │ {} 
    ───┼───┼───
     {} │ {} │ {} 
    ───┼───┼───
     {} │ {} │ {} 
    """.format(*chars))

