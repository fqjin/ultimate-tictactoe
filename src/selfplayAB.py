import numpy as np
from alphabeta import ABPlayer, RandABTree
from play import play


def selfplayAB(number, player):
    result, moves, evals = play(player, player, verbose=False)

    savepath = '../selfplayAB/' + str(number).zfill(5)

    moves.append((result, result))  # append result to keep as one array
    moves = np.asarray(moves, dtype=np.int8)
    np.save(savepath, moves)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--selfplay', action='store_true')
    parser.add_argument('--range', type=int, nargs=2)
    args = parser.parse_args()

    if args.selfplay:
        p = ABPlayer(RandABTree, max_depth=3, selfplay=True)
        for i in range(*args.range):
            print(i)
            # With batching, GPU is 4x faster than CPU, ~26 sec per game
            # Currently CPU limited, so run multiple threads in parallel
            selfplayAB(i, p)
