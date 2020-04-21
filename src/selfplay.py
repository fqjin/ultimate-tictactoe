import numpy as np
import torch
from network import UTTTNet
from net_player import NetPlayer
from play import play


def selfplay(nodes, number, model, device='cpu'):
    savelist = []
    player = NetPlayer(nodes, v_mode=True, selfplay=True,
                       model=model, device=device, noise=True,
                       savelist=savelist)
    result, moves = play(player, player, temp=(3, 0.5))

    savepath = '../selfplay/' + str(number).zfill(5)
    np.savez(savepath, result=result, moves=moves, visits=savelist)


best_net = '500_5000bs2048lr0.1d0.001e4'


if __name__ == '__main__':
    for i in range(5000, 5500):
        # CPU is still faster than GPU
        # 2.42 ms (cpu) vs 2.86 ms (cuda) : 5b x 64f
        # 7.05 ms (cpu) vs 4.43 ms (cuda) : 10b x 128f
        # Network is too small to see benefit of GPU
        # May need to implement batching (virtual MCTS).
        device = 'cpu'
        print(f'Using network {best_net}')
        m = UTTTNet()
        m.load_state_dict(torch.load(f'../models/{best_net}.pt',
                                     map_location=device))
        m = m.to(device).eval()
        selfplay(1000, i, m, device)

