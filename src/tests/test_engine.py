import pytest
from engine import *


def test_bit2board_table():
    assert len(bit2board_table) == 4 ** 9  # check for hash collisions
    assert bit2board_table[ZERO_BIT] == (0, 0, 0, 0, 0, 0, 0, 0, 0)


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
    assert result_table[ZERO_BIT] == 0
    assert get_result((2, 1, 1, 0, 1, 0, 1, 2, 2)) == 1
    assert result_table[TEST_BIT] == 1
    assert get_result((1, 2, 1, 0, 2, 1, 1, 2, 3)) == 2
    assert get_result((2, 2, 2, 0, 0, 0, 1, 1, 1)) == 3
    assert get_result((2, 1, 2, 1, 3, 1, 2, 1, 2)) == 3
    assert result_table[hash((2, 1, 2, 1, 3, 1, 2, 1, 2))] == 3


def test_move_table():
    assert full_move_table[0][0][ZERO_BIT] == hash((1, 0, 0, 0, 0, 0, 0, 0, 0))
    assert full_move_table[1][1][ZERO_BIT] == hash((0, 2, 0, 0, 0, 0, 0, 0, 0))
    assert full_move_table[2][2][ZERO_BIT] == hash((0, 0, 3, 0, 0, 0, 0, 0, 0))
    assert full_move_table[3][0][TEST_BIT] == hash((2, 1, 1, 1, 1, 0, 1, 2, 2))
    assert full_move_table[4][1][TEST_BIT] is None
    assert full_move_table[5][2][TEST_BIT] == hash((2, 1, 1, 0, 1, 3, 1, 2, 2))
    assert full_move_table[6][0][TEST_BIT] is None

