import numpy as np
from network import UTTTNet
from net_player import NetPlayer
from play import play


def selfplay(nodes, number, model, device='cpu'):
    savelist = []
    player = NetPlayer(nodes, v_mode=True, selfplay=True,
                       model=model, device=device, savelist=savelist)
    result, moves = play(player, player, verbose=True)

    savepath = '../selfplay/' + str(number).zfill(5)
    np.savez(savepath, result=result, moves=moves, visits=savelist)


if __name__ == '__main__':
    for i in range(0, 100):
        # CPU is still faster than GPU
        # 2.42 ms (cpu) vs 2.86 ms (cuda) : 5b x 64f
        # 7.05 ms (cpu) vs 4.43 ms (cuda) : 10b x 128f
        # Network is too small to see benefit of GPU
        # May need to implement batching (virtual MCTS).
        device = 'cpu'
        m = UTTTNet().to(device).eval()
        selfplay(1000, 0, m, device)

