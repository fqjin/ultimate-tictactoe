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
    m = UTTTNet()
    selfplay(100, 0, m, 'cpu')

