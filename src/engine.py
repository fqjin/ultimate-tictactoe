"""Code to run UTTT games"""
import numpy as np

decode_dict = {
    0: ' ',
    1: 'X',
    2: 'O',
    3: 'T',
}
all_boards = list(np.ndindex(*[4] * 9))

FULL_HOUSE = tuple(range(9))
ZERO_BOARD = (0,) * 9
ONES_BOARD = (1,) * 9
TWOS_BOARD = (2,) * 9
DRAW_BOARD = (3,) * 9
BOARD_LIST = (ZERO_BOARD, ONES_BOARD, TWOS_BOARD, DRAW_BOARD)
TEST_BOARD = (2, 1, 1, 0, 1, 0, 1, 2, 2)
TEST_BOARD2 = (0, 0, 1, 2, 2, 0, 0, 1, 1)


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
    """Calculates result of row

    Args:
        row: 3-tuple
        result_list: 3-list of draw_counter, p1, p2.
            List will be modified by this function.
    """
    if 3 in row:
        # draw b/c draw tile
        result_list[0] += 1
    elif 1 in row and 2 in row:
        # draw b/c both X and O tiles
        result_list[0] += 1
    elif 0 in row:
        # not win b/c empty tile
        pass
    elif row[0] == 1:
        # X wins
        result_list[1] = 1
    else:
        # O wins
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


big_result_table = {board: get_result(board) for board in all_boards}
result_table = {}
for board, result in big_result_table.items():
    if 3 in board:
        continue
    if result == 3 and 0 in board:
        # board is drawn but still has legal moves
        result_table[board] = 0
    else:
        result_table[board] = result


def make_move_table(move_index):
    """Returns mapping of boards to boards after a move"""
    # Move_table might still be useful because tuples are immutable
    table_1 = {}
    table_2 = {}
    for board, result in result_table.items():
        if result:
            continue
        if board[move_index] != 0:
            continue
        tmp = list(board)
        tmp[move_index] = 1
        table_1[board] = tuple(tmp)
        tmp[move_index] = 2
        table_2[board] = tuple(tmp)
    return table_1, table_2


full_move_table = [make_move_table(i) for i in range(9)]


legal_moves_table = {}
for board, result in result_table.items():
    if not result:
        legal_moves_table[board] = tuple(not board[i] for i in range(9))


class BigBoard:
    """UTTT Board"""
    def __init__(self, boards=(ZERO_BOARD,) * 9, mover=0, sectors=FULL_HOUSE):
        self.boards = list(boards)
        self.mover = mover
        self.sectors = sectors
        # Sacrifice drawn incomplete sector propagation
        # to eliminate need for a secret_state
        self.states = [result_table[b] for b in self.boards]
        self.result = big_result_table[tuple(self.states)]
        if not self.result:
            self.legal_moves = self.get_legal_moves()
        else:
            self.legal_moves = None

    def copy(self):
        # Directly copying attributes saves some lookups/calls
        board = BigBoard.__new__(BigBoard)
        board.boards = self.boards.copy()
        board.mover = self.mover
        board.sectors = self.sectors
        board.states = self.states.copy()
        board.result = self.result
        board.legal_moves = self.legal_moves
        return board

    def draw(self):
        """Draws 9x9 BigBoard"""
        chars = [[decode_dict[v] for v in b] for b in self.boards]
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
                else legal_moves_table[self.boards[i]] for i in range(9)]

    def move(self, sector, tile):
        if not self.legal_moves[sector][tile]:
            raise ValueError(f'Illegal move at sector {sector}, tile {tile}')
        # Update boards
        self.boards[sector] = full_move_table[tile][self.mover][self.boards[sector]]
        # Update states
        new_state = result_table[self.boards[sector]]
        if new_state:
            self.states[sector] = new_state
            self.result = big_result_table[tuple(self.states)]
        # Update mover
        self.mover = 1 - self.mover
        # Update legal sectors
        if self.states[tile]:
            self.sectors = FULL_HOUSE
        else:
            self.sectors = (tile,)
        # Update legal moves
        if not self.result:
            self.legal_moves = self.get_legal_moves()
        else:
            self.legal_moves = None
            self.sectors = ()
