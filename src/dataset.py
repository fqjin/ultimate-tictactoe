import torch
from torch.nn.functional import interpolate
from engine import BigBoard, bit2board_table


def board_to_planes(bigboard: BigBoard):
    """Converts board to tensor planes

    Eight 9x9 input planes:
    - mover, X, O, legal moves, big empty tiled, big X, big O, big T
    - empty 9x9 is NOT needed, see AlphaGoZero paper

    Return shape: (1, 8, 9, 9)
    """
    boards = [bit2board_table[b] for b in bigboard.bits]
    boards = torch.tensor(boards).view(3, 3, 3, 3)
    boards = torch.cat(boards.chunk(3, dim=0), dim=2)
    boards = torch.cat(boards.chunk(3, dim=1), dim=3)

    legal = torch.tensor(bigboard.legal_moves, dtype=torch.float32).view(3, 3, 3, 3)
    legal = torch.cat(legal.chunk(3, dim=0), dim=2)
    legal = torch.cat(legal.chunk(3, dim=1), dim=3)

    mover = torch.full((1, 1, 9, 9), bigboard.mover)
    small_x = (boards == 1).float()
    small_o = (boards == 2).float()

    states = torch.tensor(bigboard.states).view(1, 1, 3, 3)
    big_empty = (states == 0).float().repeat((1, 1, 3, 3))
    big_x = interpolate((states == 1).float(), scale_factor=3)
    big_o = interpolate((states == 2).float(), scale_factor=3)
    big_t = interpolate((states == 3).float(), scale_factor=3)

    planes = torch.cat([mover,
                        small_x,
                        small_o,
                        legal,
                        big_empty,
                        big_x,
                        big_o,
                        big_t], dim=1)
    return planes

