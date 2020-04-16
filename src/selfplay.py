import numpy as np
import torch
from network import UTTTNet
from net_player import NetPlayer
from play import play


def selfplay(nodes, number, model, device='cpu'):
    savelist = []
    player = NetPlayer(nodes, v_mode=False, selfplay=True,
                       model=model, device=device, noise=True,
                       savelist=savelist)
    result, moves = play(player, player, temp=(3, 0.5))

    savepath = '../selfplay/' + str(number).zfill(5)
    np.savez(savepath, result=result, moves=moves, visits=savelist)


if __name__ == '__main__':
    for i in range(2500, 3000):
        # CPU is still faster than GPU
        # 2.42 ms (cpu) vs 2.86 ms (cuda) : 5b x 64f
        # 7.05 ms (cpu) vs 4.43 ms (cuda) : 10b x 128f
        # Network is too small to see benefit of GPU
        # May need to implement batching (virtual MCTS).
        device = 'cpu'
        m = UTTTNet()
        m.load_state_dict(torch.load(f'../models/250_2500bs2048lr0.1d0.001e4.pt',
                                     map_location=device))
        m = m.to(device).eval()
        selfplay(1000, i, m, device)

