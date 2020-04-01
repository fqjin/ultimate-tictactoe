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


def check_row(row, result_list):
    """Gets result of row

    Args:
        row: 3-tuple
        result_list: list of draw_counter, p1, p2.
            List will be modified by this function.
    """
    if 3 in row:
        result_list[0] += 1
    elif 1 in row and 2 in row:
        result_list[0] += 1
    elif 0 in row:
        pass
    elif row[0] == 1:
        result_list[1] = 1
    else:  # row[0] == 2
        result_list[2] = 1


def get_result(board):
    """Returns win/draw/loss result of board

    Returns:
        0: undecided
        1: player 1 win
        2: player 2 win
        3: draw or illegal (both win)
    """
    result_list = [0, 0, 0]  # [draw_counter, p1, p2]

    # Check rows
    check_row(board[0:3], result_list)
    check_row(board[3:6], result_list)
    check_row(board[6:9], result_list)
    # Check columns
    check_row(board[0::3], result_list)
    check_row(board[1::3], result_list)
    check_row(board[2::3], result_list)
    # Check diags
    check_row(board[0::4], result_list)
    check_row(board[2:7:2], result_list)

    if result_list[0] == 8:
        return 3
    return result_list[1] + 2 * result_list[2]


result_table = {bit: get_result(board)
                for bit, board in bit2board_table.items()}


def make_move_table(move_index):
    """Returns mapping of bits to bits after move"""
    table_1 = {}
    table_2 = {}
    table_3 = {}
    for bit, board in bit2board_table.items():
        if board[move_index] != 0:
            table_1[bit] = None
            table_2[bit] = None
            table_3[bit] = None
        else:
            tmp = list(board)
            tmp[move_index] = 1
            table_1[bit] = hash(tuple(tmp))
            tmp[move_index] = 2
            table_2[bit] = hash(tuple(tmp))
            tmp[move_index] = 3
            table_3[bit] = hash(tuple(tmp))
    return table_1, table_2, table_3


full_move_table = [make_move_table(i) for i in range(9)]


class BigBoard:
    """UTTT Board"""
    def __init__(self, bits=None):
        if not bits:
            self.bits = [ZERO_BIT] * 9
        else:
            self.bits = bits

    def draw(self):
        boards = [bit2board_table[b] for b in self.bits]
        states = [result_table[b] for b in self.bits]
        for i, state in enumerate(states):
            if state != 0:
                boards[i] = (state, ) * 9
        chars = [[decode_dict[v] for v in board] for board in boards]
        rows = [chars[3*i][3*j:3*j+3] +
                chars[3*i+1][3*j:3*j+3] +
                chars[3*i+2][3*j:3*j+3] for i, j in np.ndindex(3, 3)]
        chars = [c for row in rows for c in row]
        print("""
         {} │ {} │ {} ┃ {} │ {} │ {} ┃ {} │ {} │ {} 
        ───┼───┼───╋───┼───┼───╋───┼───┼───
         {} │ {} │ {} ┃ {} │ {} │ {} ┃ {} │ {} │ {} 
        ───┼───┼───╋───┼───┼───╋───┼───┼───
         {} │ {} │ {} ┃ {} │ {} │ {} ┃ {} │ {} │ {} 
        ━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━
         {} │ {} │ {} ┃ {} │ {} │ {} ┃ {} │ {} │ {} 
        ───┼───┼───╋───┼───┼───╋───┼───┼───
         {} │ {} │ {} ┃ {} │ {} │ {} ┃ {} │ {} │ {} 
        ───┼───┼───╋───┼───┼───╋───┼───┼───
         {} │ {} │ {} ┃ {} │ {} │ {} ┃ {} │ {} │ {} 
        ━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━
         {} │ {} │ {} ┃ {} │ {} │ {} ┃ {} │ {} │ {} 
        ───┼───┼───╋───┼───┼───╋───┼───┼───
         {} │ {} │ {} ┃ {} │ {} │ {} ┃ {} │ {} │ {} 
        ───┼───┼───╋───┼───┼───╋───┼───┼───
         {} │ {} │ {} ┃ {} │ {} │ {} ┃ {} │ {} │ {}         
        """.format(*chars))
