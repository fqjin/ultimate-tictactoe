import numpy as np
import torch
from network import UTTTNet
from net_player import BatchNetPlayer
from play import play


def selfplay(nodes, number, model, device='cpu'):
    savelist = []
    player = BatchNetPlayer(nodes, v_mode=True, selfplay=True,
                            model=model, device=device, noise=True,
                            savelist=savelist)
    result, moves, evals = play(player, player, temp=(7, 1.0))

    savepath = '../selfplay/' + str(number).zfill(5)
    np.savez(savepath, result=result, moves=moves, visits=savelist, evals=evals)


best_net = '2400_24000bs2048lr0.1d0.001e6'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--range', type=int, nargs=2, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    print(f'Using network {best_net}')
    m = UTTTNet()
    m.load_state_dict(torch.load(f'../models/{best_net}.pt',
                                 map_location=device))
    m = m.to(device).eval()
    # TODO: write a load_model function in network.py
    for i in range(*args.range):
        # With batching, GPU is 4x faster than CPU, ~26 sec per game
        # Currently CPU limited, so run multiple threads in parallel
        selfplay(1000, i, m, device)

