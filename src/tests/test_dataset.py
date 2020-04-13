import pytest
import torch
from dataset import board_to_planes
from engine import BigBoard


def test_board_to_planes():
    a = BigBoard()
    a.move(0, 4)
    a.move(4, 8)
    a.move(8, 8)
    a.states[1] = 1
    a.states[2] = 2
    a.states[3] = 3
    planes = board_to_planes(a)

    assert planes.shape == (1, 8, 9, 9)
    assert planes[0, 0, 2, 4] == -1  # mover O
    assert planes[0, 1].sum() == 2  # two X moves
    assert planes[0, 1, 1, 1] == 1  # X @ (0, 4)
    assert planes[0, 1, 8, 8] == 1  # X @ (8, 8)
    assert planes[0, 2].sum() == 1  # one O move
    assert planes[0, 2, 5, 5] == 1  # O @ (4, 8)
    assert planes[0, 3].sum() == 8  # 8 legal moves
    assert planes[0, 3, 6, 7] == 1  # a legal move
    assert torch.all(torch.eq(planes[0, 4, 0:3, 3:6], planes[0, 4, 3:6, 6:9]))
    assert planes[0, 4, 0, 0] == 1  # state is empty
    assert planes[0, 4, 0, 1] == 0  # state is filled
    assert planes[0, 5, 1, 4] == 1  # big X
    assert planes[0, 6, 1, 8] == 1  # big O
    assert planes[0, 7, 5, 1] == 1  # big T

