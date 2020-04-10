import numpy as np
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from engine import BigBoard, bit2board_table
from tree import value_dict


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


def game_to_data(game):
    """Plays out the stored game and returns planes, policy, result"""
    planes = []
    b = BigBoard()
    planes.append(board_to_planes(b))
    for m in game['moves'][:-1]:
        b.move(*m)
        planes.append(board_to_planes(b))

    policy = torch.from_numpy(game['visits']).float()
    policy /= torch.sum(policy, dim=(1, 2), keepdim=True)
    policy = policy.view(-1, 3, 3, 3, 3)
    policy = torch.cat(policy.chunk(3, dim=1), dim=3)
    policy = torch.cat(policy.chunk(3, dim=2), dim=4)
    policy = policy.squeeze_()

    result = [value_dict[game['result'].item()]] * len(planes)
    return planes, policy, result


class GameDataset(Dataset):
    """Loads selfplay games"""
    def __init__(self, start, end, device='cpu'):
        path = '../selfplay/'
        self.planes = []
        self.policy = []
        self.result = []
        for i in range(start, end):
            game = np.load(path + str(i).zfill(5) + '.npz')
            x, p, v = game_to_data(game)
            self.planes.extend(x)
            self.policy.extend(p)
            self.result.extend(v)
        self.planes = torch.cat(self.planes, dim=0).to(device)
        self.policy = torch.stack(self.policy).to(device)
        self.result = torch.tensor(self.result,
                                   dtype=torch.float32,
                                   device=device)
        # TODO: Mixed/half precision training

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        # TODO: flip/rot augmentation
        return self.planes[idx], (self.policy[idx], self.result[idx])
