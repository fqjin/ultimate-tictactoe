import torch
from stats import make_stats
from network import UTTTNet
from net_player import NetPlayer


def model_vs_model(weights1, weights2, device='cpu', nodes=100, ):
    m1 = UTTTNet()
    m2 = UTTTNet()
    m1.load_state_dict(torch.load(f'../models/{weights1}.pt', map_location=device))
    m2.load_state_dict(torch.load(f'../models/{weights2}.pt', map_location=device))
    m1.to(device).eval()
    m2.to(device).eval()
    p1 = NetPlayer(nodes=nodes, v_mode=True, model=m1, device=device, noise=False)
    p2 = NetPlayer(nodes=nodes, v_mode=True, model=m2, device=device, noise=False)
    make_stats(p1, p2, weights1, weights2, weights1+'_vs_'+weights2, num=50, temp=(3, 0.5))


def model_VN(weights1, device='cpu', nodes=100, ):
    m = UTTTNet()
    m.load_state_dict(torch.load(f'../models/{weights1}.pt', map_location=device))
    m.to(device).eval()
    p1 = NetPlayer(nodes=nodes, v_mode=True, model=m, device=device, noise=False)
    p2 = NetPlayer(nodes=nodes, v_mode=False, model=m, device=device, noise=False)
    make_stats(p1, p2, 'Vmode', 'Nmode', weights1+'_V_vs_N', num=50, temp=(3, 0.5))


if __name__ == '__main__':
    base = '00000'
    net1 = 'newplane_from00000_100_1000bs2048lr0.1d0.001e3'
    net2 = 'from00000_150_1500bs2048lr0.1d0.001e3'
    net3 = 'scr_dl_200_2000bs2048lr0.1d0.001e3'
    net4 = '250_2500bs2048lr0.1d0.001e4'
    net5 = '300_3000bs2048lr0.1d0.001e4'
    net6 = '350_3500bs2048lr0.1d0.001e5'
    new = net6
    model_VN(new)
    # model_vs_model(base, new)
    # model_vs_model(net1, new)
    model_vs_model(net2, new)
    model_vs_model(net3, new)
    model_vs_model(net4, new)
    model_vs_model(net5, new)
