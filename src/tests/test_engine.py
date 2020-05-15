import pytest
from engine import *


def test_check_row():
    x = [0, 0, 0]
    check_row((0, 0, 0), x)
    assert x == [0, 0, 0]
    check_row((1, 1, 0), x)
    assert x == [0, 0, 0]
    check_row((0, 2, 1), x)
    assert x == [1, 0, 0]
    check_row((0, 3, 0), x)
    assert x == [2, 0, 0]
    check_row((1, 1, 1), x)
    assert x == [2, 1, 0]
    check_row((2, 2, 2), x)
    assert x == [2, 1, 1]


def test_get_result():
    assert get_result((0,) * 9) == 0
    assert result_table[ZERO_BOARD] == 0
    assert get_result((2, 1, 1, 0, 1, 0, 1, 2, 2)) == 1
    assert result_table[TEST_BOARD] == 1
    assert get_result((1, 2, 1, 0, 2, 1, 1, 2, 3)) == 2
    assert get_result((2, 2, 2, 0, 0, 0, 1, 1, 1)) == 3
    assert get_result((2, 1, 2, 1, 3, 1, 2, 1, 2)) == 3
    assert big_result_table[(2, 1, 1,
                             1, 2, 2,
                             0, 2, 1)] == 3
    assert result_table[(2, 1, 1,
                         1, 2, 2,
                         0, 2, 1)] == 0
    assert result_table[(2, 1, 1,
                         1, 2, 2,
                         1, 2, 1)] == 3


def test_move_table():
    assert full_move_table[0][0][ZERO_BOARD] == (1, 0, 0, 0, 0, 0, 0, 0, 0)
    assert full_move_table[1][1][ZERO_BOARD] == (0, 2, 0, 0, 0, 0, 0, 0, 0)
    assert full_move_table[1][0][TEST_BOARD2] == (0, 1, 1, 2, 2, 0, 0, 1, 1)
    with pytest.raises(KeyError):
        x = full_move_table[3][1][TEST_BOARD2]


def test_legal_moves_table():
    assert legal_moves_table[ZERO_BOARD] == (1,) * 9
    assert legal_moves_table[TEST_BOARD2] == (1, 1, 0, 0, 0, 1, 1, 0, 0)
    with pytest.raises(KeyError):
        x = legal_moves_table[TEST_BOARD]


def test_get_legal_moves():
    bb = BigBoard(sectors=[1, 4])
    legal_moves = [ZERO_BOARD] * 9
    legal_moves[1] = ONES_BOARD
    legal_moves[4] = ONES_BOARD
    assert bb.legal_moves == legal_moves

    legal_moves = [ONES_BOARD] * 9
    legal_moves[5] = ZERO_BOARD
    legal_moves[7] = (1, 1, 0, 0, 0, 1, 1, 0, 0)
    bb = BigBoard(boards=(ZERO_BOARD, ZERO_BOARD, ZERO_BOARD,
                          ZERO_BOARD, ZERO_BOARD, TEST_BOARD,
                          ZERO_BOARD, TEST_BOARD2, ZERO_BOARD))
    assert bb.legal_moves == legal_moves


def test_move():
    bb = BigBoard()
    bb.move(sector=1, tile=3)
    legal_moves = [ZERO_BOARD] * 9
    legal_moves[3] = ONES_BOARD
    assert bb.mover == 1
    assert bb.sectors == (3,)
    assert bb.legal_moves == legal_moves
    assert bb.boards == [ZERO_BOARD, (0, 0, 0, 1, 0, 0, 0, 0, 0), ZERO_BOARD,
                         ZERO_BOARD, ZERO_BOARD, ZERO_BOARD,
                         ZERO_BOARD, ZERO_BOARD, ZERO_BOARD]

    bb.move(sector=3, tile=1)
    assert bb.mover == 0
    assert bb.sectors == (1,)
    assert bb.boards == [ZERO_BOARD, (0, 0, 0, 1, 0, 0, 0, 0, 0), ZERO_BOARD,
                         (0, 2, 0, 0, 0, 0, 0, 0, 0), ZERO_BOARD, ZERO_BOARD,
                         ZERO_BOARD, ZERO_BOARD, ZERO_BOARD]

    with pytest.raises(ValueError):
        bb.move(sector=1, tile=3)
