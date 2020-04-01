"""Code to run UTTT games"""
import numpy as np

decode_dict = {
    0: ' ',
    1: 'X',
    2: 'O',
    3: 'T',
}
bit2board_table = {hash(indices): indices
                   for indices in np.ndindex(*[4] * 9)}
ZERO_BIT = hash((0,) * 9)
TEST_BIT = hash((2, 1, 1, 0, 1, 0, 1, 2, 2))


def draw_board(board):
    chars = [decode_dict[v] for v in board]
    print("""
     {} │ {} │ {} 
    ───┼───┼───
     {} │ {} │ {} 
    ───┼───┼───
     {} │ {} │ {} 
    """.format(*chars))

