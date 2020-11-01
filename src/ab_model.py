import numpy as np
import torch
from network import ABNet
from alphabeta import ABTree, value_dict
from dataset import board_to_planes


def load_ABnet(weights, device=None, **kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    print(f'Using network {weights}')
    m = ABNet(**kwargs)
    m.load_state_dict(torch.load(f'../models/{weights}.pt',
                                 map_location=device))
    m = m.to(device).eval()
    return m


class NetABTree(ABTree):
    def __init__(self, board, alpha, beta, noise=True,
                 model=None, device=None):
        super().__init__(board, alpha, beta, noise)
        if model is None or device is None:
            raise TypeError('model or device argument is missing')
        self.model = model
        self.device = device
        self.kwargs = {'model': model, 'device': device}

    def eval_fn(self):
        return None

    def init_children(self):
        super().init_children()
        non_terminal = []
        for c in self.children:
            child = c[2]
            if child.board.result:
                child.value = value_dict[child.board.result]
            else:
                non_terminal.append(child)
        x = torch.cat([board_to_planes(child.board) for child in non_terminal])
        with torch.no_grad():
            values = self.model(x.to(self.device))
        if self.noise:
            values += 0.1 * (torch.rand_like(values) - 0.5)
        for v, child in zip(values, non_terminal):
            child.value = v.item()
