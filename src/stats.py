"""Play 1000 games, write to pgn, calculate ordo stats"""
from tqdm import tqdm
from play import play
from players import *

result_strings = {
    1: '[Result "1-0"]\n1-0\n',
    2: '[Result "0-1"]\n0-1\n',
    3: '[Result "1/2-1/2"]\n1/2-1/2\n',
}


def make_stats(p1, p2, name1, name2, out_name, num=500, kt1=True, kt2=True, temp=None, save=False):
    """Play games between two players and save stats

    P1 and P2 will play `num` rounds (two games as X and O)
    Results will be saved as PGN file in ordo

    Args:
        p1: BasePlayer object
        p2: BasePlayer object
        name1: name of player 1
        name2: name of player 2
        out_name: output pgn file name
        num: number of rounds
        kt1: keep_tree for player 1
        kt2: keep_tree for player 2
        temp: tuple of moves and invtemp. Default None
        save: save games
    """
    with open(f'../ordo/{out_name}.pgn', '+w') as pgn:
        for i in tqdm(range(num)):
            result, moves, evals = play(p1, p2,
                                        verbose=False,
                                        press_enter=False,
                                        give_moves0=kt1,
                                        give_moves1=kt2,
                                        temp=temp)
            pgn.write(f'[White "{name1}"]\n')
            pgn.write(f'[Black "{name2}"]\n')
            pgn.write(result_strings[result])
            if save:
                np.savez(f'../games/{out_name}_game{i}a',
                         result=result, moves=moves, evals=evals)

            result, moves, evals = play(p2, p1,
                                        verbose=False,
                                        press_enter=False,
                                        give_moves0=kt2,
                                        give_moves1=kt1,
                                        temp=temp)
            pgn.write(f'[White "{name2}"]\n')
            pgn.write(f'[Black "{name1}"]\n')
            pgn.write(result_strings[result])
            if save:
                np.savez(f'../games/{out_name}_game{i}b',
                         result=result, moves=moves, evals=evals)
