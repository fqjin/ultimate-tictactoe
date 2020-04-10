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
    p1 = NetPlayer(nodes=nodes, v_mode=True, model=m1, device=device)
    p2 = NetPlayer(nodes=nodes, v_mode=False, model=m2, device=device)
    make_stats(p1, p2, 'Vmode', 'Nmode', weights1+'_V_vs_N', num=5, kt1=True, kt2=True)


if __name__ == '__main__':
    base = '00000'
    net1 = 'from00000_0_100bs1024lr0.1d1e-05e10'
    net2 = 'from00000_0_100bs1024lr0.1d1e-05e10'
    # model_vs_model(base, net1)
    # model_vs_model(base, net2)
    model_vs_model(net1, net2)
