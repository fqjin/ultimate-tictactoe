import torch
from stats import make_stats
from ab_model import load_ABnet, NetABTree
from alphabeta import ABPlayer


def model_vs_model(weights1, weights2, device='cpu',
                   depth=3, num=50, temp=(6, 0.5),
                   save=False):
    m1 = load_ABnet(weights1, device)
    m2 = load_ABnet(weights2, device)
    p1 = ABPlayer(NetABTree, max_depth=depth, model=m1, device=device)
    p2 = ABPlayer(NetABTree, max_depth=depth, model=m2, device=device)
    make_stats(p1, p2, weights1, weights2, weights1+'_vs_'+weights2,
               num=num, temp=temp, save=save)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', type=int, required=True)
    parser.add_argument('--e', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nets = [
        'init_ab',
        '1000_10000bs2048lr0.1d0.001abe5',
        '1000_10000bs2048lr0.1d0.001abe10',
    ]
    if args.e:
        nets[-1] = nets[-1][:-1] + str(args.e)
    if args.flag == 0:
        model_vs_model(nets[-2], nets[-1], device=device)
    elif args.flag == 1:
        model_vs_model(nets[-2], nets[0], device=device)

