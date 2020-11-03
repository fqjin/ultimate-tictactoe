nodes = 4000
depth = 5
ab_games = 40
print(f'{nodes} nodes vs depth {depth}')

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from selfplay import *
m0 = load_model()
p0 = BatchNetPlayer(nodes, v_mode=True, selfplay=False,
                             model=m0, device=device, noise=False)

from selfplayAB import *
m1 = load_ABnet(best_net, device=device)
p1 = ABPlayer(NetABTree, max_depth=depth, selfplay=False,
              model=m1, device=device)

from play import play
r, m, e, t = play(p0, p1, verbose=True, temp=None, timing=True)
np.savez(f'../games/net40k_{nodes}n_vs_AB{ab_games}k_d{depth}a',
         result=r, moves=m, evals=e, times=t)
t0a = np.sum(t[::2])
t1a = np.sum(t[1::2])

r, m, e, t = play(p1, p0, verbose=True, temp=None, timing=True)
np.savez(f'../games/net40k_{nodes}n_vs_AB{ab_games}k_d{depth}b',
         result=r, moves=m, evals=e, times=t)
t1b = np.sum(t[::2])
t0b = np.sum(t[1::2])

print('Timings a:', t0a, t1a, t0a/t1a)
print('Timings b:', t0b, t1b, t0b/t1b)
print('Timings x:', t0a+t0b, t1a+t1b, (t0a+t0b)/(t1a+t1b))
