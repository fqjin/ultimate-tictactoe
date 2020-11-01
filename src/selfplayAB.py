import numpy as np
import torch
from alphabeta import ABPlayer, RandABTree
from ab_model import NetABTree, load_ABnet
from play import play


best_net = '1000_10000bs2048lr0.1d0.001abe5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def selfplayAB(number, player):
    result, moves, evals = play(player, player, temp=(7, 1.0))

    savepath = '../selfplayAB/' + str(number).zfill(5)

    moves.append((result, result))  # append result to keep as one array
    moves = np.asarray(moves, dtype=np.int8)
    np.save(savepath, moves)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--range', type=int, nargs=2, required=True)
    args = parser.parse_args()

    # p = ABPlayer(RandABTree, max_depth=3, selfplay=True)

    model = load_ABnet(best_net, device=device)
    p = ABPlayer(NetABTree, max_depth=3, selfplay=True,
                 model=model, device=device)
    for i in range(*args.range):
        print(i)
        # With batching, GPU is 4x faster than CPU, ~26 sec per game
        # Currently CPU limited, so run multiple threads in parallel
        selfplayAB(i, p)
