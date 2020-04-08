"""Code to run UTTT games"""
import numpy as np

decode_dict = {
    0: ' ',
    1: 'X',
    2: 'O',
    3: 'T',
}
# TODO: Check if using tuples as keys rather than its hash is faster
#  because the dictionary is presumably hashing its keys anyway!
# 175 ns to hash board and access bit2board
# 122 ns to access board2board without hash
# Doesn't make big impact on final speed, but would simplify the code.
bit2board_table = {hash(indices): indices
                   for indices in np.ndindex(*[4] * 9)}

FULL_HOUSE = tuple(range(9))
ZERO_BOARD = (0,) * 9
ONES_BOARD = (1,) * 9
ZERO_BIT = hash(ZERO_BOARD)
ONES_BIT = hash(ONES_BOARD)
TWOS_BIT = hash((2,) * 9)
DRAW_BIT = hash((3,) * 9)
BIT_LIST = (ZERO_BIT, ONES_BIT, TWOS_BIT, DRAW_BIT)
TEST_BIT = hash((2, 1, 1, 0, 1, 0, 1, 2, 2))
TEST_BIT2 = hash((0, 0, 1, 2, 2, 0, 0, 1, 1))


def draw_board(board):
    chars = [decode_dict[v] for v in board]
    print('''\
     {} │ {} │ {} 
    ───┼───┼───
     {} │ {} │ {} 
    ───┼───┼───
     {} │ {} │ {} \
    '''.format(*chars))


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


big_result_table = {bit: get_result(board)
                    for bit, board in bit2board_table.items()}
result_table = {}
for bit, board in bit2board_table.items():
    if 3 in board:
        continue
    result = big_result_table[bit]
    if result == 3 and 0 in board:
        result_table[bit] = 0
    else:
        result_table[bit] = result


def make_move_table(move_index):
    """Returns mapping of bits to bits after move"""
    table_1 = {}
    table_2 = {}
    for bit, board in bit2board_table.items():
        if 3 in board:
            continue
        result = big_result_table[bit]
        if result and not (result == 3 and 0 in board):
            continue
        if board[move_index] != 0:
            continue
        else:
            tmp = list(board)
            tmp[move_index] = 1
            table_1[bit] = hash(tuple(tmp))
            tmp[move_index] = 2
            table_2[bit] = hash(tuple(tmp))
    return table_1, table_2


full_move_table = [make_move_table(i) for i in range(9)]


legal_moves_table = {}
for bit, board in bit2board_table.items():
    if 3 in board:
        continue
    if not result_table[bit]:
        legal_moves_table[bit] = tuple(not board[i] for i in range(9))


class BigBoard:
    """UTTT Board"""
    def __init__(self, bits=(ZERO_BIT,)*9, mover=0, sectors=FULL_HOUSE):
        self.bits = list(bits)
        self.mover = mover
        self.sectors = sectors
        # Sacrifice drawn incomplete sector propagation
        # to eliminate need for a secret_state
        self.states = [result_table[b] for b in self.bits]
        self.result = big_result_table[hash(tuple(self.states))]
        if not self.result:
            self.legal_moves = self.get_legal_moves()
        else:
            self.legal_moves = None

    def copy(self):
        # Directly copying attributes saves some function calls
        board = BigBoard.__new__(BigBoard)
        board.bits = self.bits.copy()
        board.mover = self.mover
        board.sectors = self.sectors
        board.states = self.states.copy()
        board.result = self.result
        board.legal_moves = self.legal_moves
        return board

    def draw(self):
        """Draws 9x9 BigBoard"""
        boards = [bit2board_table[b] for b in self.bits]
        chars = [[decode_dict[v] for v in board] for board in boards]
        rows = [chars[3*i][3*j:3*j+3] +
                chars[3*i+1][3*j:3*j+3] +
                chars[3*i+2][3*j:3*j+3] for i, j in np.ndindex(3, 3)]
        chars = [c for row in rows for c in row]
        print('''\
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
     {} │ {} │ {} ┃ {} │ {} │ {} ┃ {} │ {} │ {} \
        '''.format(*chars))
        print(f'Mover: {decode_dict[self.mover + 1]}, sector: {self.sectors}')
        draw_board(self.states)

    def get_legal_moves(self):
        """Returns list of tuples of ones and zeros describing legal moves"""
        return [ZERO_BOARD if i not in self.sectors or self.states[i]
                else legal_moves_table[self.bits[i]] for i in range(9)]

    def move(self, sector, tile):
        """Mover places tile at sector and tile location"""
        if not self.legal_moves[sector][tile]:
            raise ValueError(f'Illegal move at sector {sector}, tile {tile}')
        # Update bits
        self.bits[sector] = full_move_table[tile][self.mover][self.bits[sector]]
        # Update states
        new_state = result_table[self.bits[sector]]
        if new_state:
            self.states[sector] = new_state
            self.result = big_result_table[hash(tuple(self.states))]
        # Update other attributes
        self.mover = 1 - self.mover
        if self.states[tile]:
            # Sectors describes which sectors are allowed
            # Legal moves then removes sections with determined state
            self.sectors = FULL_HOUSE
        else:
            self.sectors = (tile,)
        # Update legal moves
        if not self.result:
            self.legal_moves = self.get_legal_moves()
        else:
            self.legal_moves = None
