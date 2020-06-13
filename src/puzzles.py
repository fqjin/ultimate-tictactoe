import glob
import numpy as np
from engine import BigBoard
from gui import GuiPlayer
from net_player import NetPlayer
from play import play
from selfplay import load_model, device


def win_in_puzzle(name, win_in, model, device='cpu', atol=1e-3):
    data = np.load(name)
    if data['result'] not in (1, 2):
        return
    if 'evals' not in data:
        return
    winner = 3 - 2 * data['result']
    moves = data['moves'].tolist()
    moves = [tuple(m) for m in moves]
    win_in = np.random.choice(win_in)
    index = len(moves) - 2 * win_in + 1
    if not np.isclose(data['evals'][index], winner, atol=atol):
        return
    moves = moves[:index]
    game = BigBoard()
    for m in moves:
        game.move(*m)
    # TODO: Random D4 rotation/reflection

    pg = GuiPlayer(x=600)
    while True:
        if winner == 1:
            pn = NetPlayer(nodes=1000, v_mode=True, model=model,
                           device=device, noise=False)
            result, _, _ = play(pg, pn, game.copy(), moves=moves.copy())
        else:
            pn = NetPlayer(nodes=1000, v_mode=True, model=model,
                           device=device, noise=False)
            result, _, _ = play(pn, pg, game.copy(), moves=moves.copy())
        if result == data['result']:
            break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--win_in', type=int, nargs='*', default=2,
                        help='Win in x moves. Default 2. nargs = *')
    parser.add_argument('--include', type=int, default=2000,
                        help='How many of the latest games to use.')
    args = parser.parse_args()

    if isinstance(args.win_in, int):
        args.win_in = [args.win_in]
    print(f'win-in: {args.win_in}')

    selfplay_games = sorted(glob.glob('../selfplay/*.npz'))
    selfplay_games = selfplay_games[-args.include:]
    np.random.shuffle(selfplay_games)

    model = load_model()
    for name in selfplay_games:
        print(name)
        win_in_puzzle(name, args.win_in, model, device)
