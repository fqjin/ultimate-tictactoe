import numpy as np
import torch
from torch.utils.data import Dataset
from random import getrandbits
from tqdm import tqdm
from engine import BigBoard


def board_to_nnue(bigboard: BigBoard):
    """Converts board to NNUE binary feature vector

    - All flattened, so no need to tile from index to spatial
    - legal moves -x- mover (separate movers)         [81 x 2]
    - small: contains X, contains O                   [81 x 2]
    - big: empty, contains X, contains O, contains T   [9 x 4]
                                          Total features: 360

    """
    features = torch.zeros(360)

    legal = torch.tensor(bigboard.legal_moves).view(-1)
    if not bigboard.mover:  # X to move
        features[0:81] = legal
    else:  # O to move
        features[81:162] = legal

    boards = torch.tensor(bigboard.boards).view(-1)
    features[162:243] = (boards == 1)  # small X
    features[243:324] = (boards == 2)  # small O

    states = torch.tensor(bigboard.states)
    features[324:333] = (states == 0)  # empty
    features[333:342] = (states == 1)  # big X
    features[342:351] = (states == 2)  # big O
    features[351:360] = (states == 3)  # big T

    return features


def board_to_nnue_batch(legals, movers, boards, states):
    features = torch.zeros(len(movers), 360)

    legals = torch.tensor(legals).view(-1, 81)
    movers = torch.tensor(movers).view(-1, 1)
    features[:, 0:81] = legals * (1-movers)
    features[:, 81:162] = legals * movers

    boards = torch.tensor(boards).view(-1, 81)
    features[:, 162:243] = (boards == 1)  # small X
    features[:, 243:324] = (boards == 2)  # small O

    states = torch.tensor(states).view(-1, 9)
    features[:, 324:333] = (states == 0)  # empty
    features[:, 333:342] = (states == 1)  # big X
    features[:, 342:351] = (states == 2)  # big O
    features[:, 351:360] = (states == 3)  # big T

    return features


def viz_feat(features):
    """Visualize binary feature vector"""
    import matplotlib.pyplot as plt
    for i in range(4):
        plt.subplot(2, 3, i+1 + (i>1))
        planes = features[81*i:81*(i+1)]
        planes = planes.view(3, 3, 3, 3)
        planes = torch.cat(planes.chunk(3, dim=0), dim=2)
        planes = torch.cat(planes.chunk(3, dim=1), dim=3)
        plt.imshow(planes[0, 0], vmin=0, vmax=1)
        plt.axis('off')

    subplot_inds = [
        [5, 3, 6],
        [5, 6, 17],
        [5, 6, 18],
        [5, 3, 12]
    ]
    for i in range(4):
        plt.subplot(*subplot_inds[i])
        stages = features[324+9*i:324+9*(i+1)].view(3, 3)
        plt.imshow(stages, vmin=0, vmax=1)
        plt.axis('off')

    plt.show()


def make_transform_dict():
    """Pre-computed indices for transforming binary feature
    vector with flips and rotations. Note that this method
    actually produces the inverse transform.
    """
    transform_dict = {}
    planes = torch.arange(81).view(3, 3, 3, 3)
    planes = torch.cat(planes.chunk(3, dim=0), dim=2)
    planes = torch.cat(planes.chunk(3, dim=1), dim=3)
    planes = planes[0, :, :, :]
    states = torch.arange(9).view(1, 3, 3)
    for aug in range(8):
        augplanes = planes.clone()
        augstates = states.clone()
        if aug & 3 == 1:
            augplanes = torch.flip(augplanes, dims=(1,))
            augstates = torch.flip(augstates, dims=(1,))
        elif aug & 3 == 2:
            augplanes = torch.flip(augplanes, dims=(2,))
            augstates = torch.flip(augstates, dims=(2,))
        elif aug & 3 == 3:
            augplanes = torch.flip(augplanes, dims=(1, 2))
            augstates = torch.flip(augstates, dims=(1, 2))
        if aug & 4:
            augplanes = torch.transpose(augplanes, 1, 2)
            augstates = torch.transpose(augstates, 1, 2)

        augplanes = augplanes.view(1, 1, 9, 9)
        augplanes = torch.cat(augplanes.chunk(3, dim=2), dim=0)
        augplanes = torch.cat(augplanes.chunk(3, dim=3), dim=1)
        augplanes = augplanes.view(81)
        augstates = augstates.reshape(9)

        transform_dict[aug] = torch.cat([
            augplanes,
            augplanes + 81,
            augplanes + 81*2,
            augplanes + 81*3,
            augstates + 81*4,
            augstates + 81*4 + 9,
            augstates + 81*4 + 9*2,
            augstates + 81*4 + 9*3,
        ])

    return transform_dict


class NNUEGameDataset(Dataset):
    """Loads selfplay games for NNUE training"""
    def __init__(self, start, end,
                 path='../selfplayNNUE/data.zip',
                 device='cpu',
                 augment=False):
        self.path = path
        self.augment = augment
        self.planes = []
        self.result = []

        WDL_dict = {
            1: 0,  # Win
            3: 1,  # Draw
            2: 2,  # Loss
        }
        xxx = np.load(path)
        for i in tqdm(range(start, end)):
            try:
                moves = xxx[str(i).zfill(5)]
            except KeyError:
                raise KeyError('Selfplay game number not in selfplay zip')

            legals = []
            movers = []
            boards = []
            states = []
            b = BigBoard()
            for m in moves[:-1]:
                legals.append(b.legal_moves)
                movers.append(b.mover)
                boards.append(b.boards.copy())
                states.append(b.states.copy())
                b.move(*m)
            planes = board_to_nnue_batch(legals, movers, boards, states)
            result = [WDL_dict[moves[-1, 0].item()]] * len(planes)

            self.planes.append(planes)
            self.result.extend(result)

        self.planes = torch.cat(self.planes, dim=0).to(device)
        self.result = torch.tensor(self.result,
                                   dtype=torch.int64,
                                   device=device)

        self.transform_dict = make_transform_dict()

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        if self.augment:
            planes = self.planes[idx]
            aug = getrandbits(3)
            if aug:  # TODO: walrus
                planes = planes[self.transform_dict[aug]]
            return planes, self.result[idx]
        else:
            return self.planes[idx], self.result[idx]
