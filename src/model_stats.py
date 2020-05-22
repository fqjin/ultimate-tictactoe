import torch
from stats import make_stats
from network import UTTTNet
from net_player import BatchNetPlayer


def model_vs_model(weights1, weights2, device='cpu',
                   nodes=100, num=50, temp=(3, 0.5),
                   save=False):
    m1 = UTTTNet()
    m2 = UTTTNet()
    m1.load_state_dict(torch.load(f'../models/{weights1}.pt', map_location=device))
    m2.load_state_dict(torch.load(f'../models/{weights2}.pt', map_location=device))
    m1.to(device).eval()
    m2.to(device).eval()
    p1 = BatchNetPlayer(nodes=nodes, v_mode=True, model=m1, device=device, noise=False)
    p2 = BatchNetPlayer(nodes=nodes, v_mode=True, model=m2, device=device, noise=False)
    make_stats(p1, p2, weights1, weights2, weights1+'_vs_'+weights2,
               num=num, temp=temp, save=save)


def model_VN(weights1, device='cpu', nodes=100):
    m = UTTTNet()
    m.load_state_dict(torch.load(f'../models/{weights1}.pt', map_location=device))
    m.to(device).eval()
    p1 = BatchNetPlayer(nodes=nodes, v_mode=True, model=m, device=device, noise=False)
    p2 = BatchNetPlayer(nodes=nodes, v_mode=False, model=m, device=device, noise=False)
    make_stats(p1, p2, 'Vmode', 'Nmode', weights1+'_V_vs_N', num=50, temp=(3, 0.5))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--flag', type=int, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    nets = [
        '00000',
        'newplane_from00000_100_1000bs2048lr0.1d0.001e3',
        'from00000_150_1500bs2048lr0.1d0.001e3',
        'scr_dl_200_2000bs2048lr0.1d0.001e3',
        '250_2500bs2048lr0.1d0.001e4',
        '300_3000bs2048lr0.1d0.001e4',
        '350_3500bs2048lr0.1d0.001e5',
        '400_4000bs2048lr0.1d0.001e4',
        '450_4500bs2048lr0.1d0.001e4',
        '500_5000bs2048lr0.1d0.001e4',
        '600_6000bs2048lr0.1d0.001e4',
        '700_7000bs2048lr0.1d0.001e4',
        '800_8000bs2048lr0.1d0.001e4',
        '900_9000bs2048lr0.1d0.001e5',
        '1000_10000bs2048lr0.1d0.001e5',
        '1100_11000bs2048lr0.1d0.001e5',
        '1200_12000bs2048lr0.1d0.001e5',
        '1300_13000bs2048lr0.1d0.001e5',
        '1400_14000bs2048lr0.1d0.001e6',
        '1500_15000bs2048lr0.1d0.001e7',
        '1600_16000bs2048lr0.1d0.001e6',
    ]
    if args.flag == 0:
        model_vs_model(nets[-5], nets[-1], device=device)
    elif args.flag == 1:
        model_vs_model(nets[-4], nets[-1], device=device)
    elif args.flag == 2:
        model_vs_model(nets[-3], nets[-1], device=device)
    elif args.flag == 3:
        model_vs_model(nets[-2], nets[-1], device=device)
    elif args.flag == 4:
        model_VN(nets[-1], device=device)
    elif args.flag == 5:
        model_vs_model(nets[-1], nets[-11], device=device,
                       num=1, nodes=10000, temp=(2, 0.5), save=True)
    else:
        raise ValueError(f'args.flag is {args.flag}')
