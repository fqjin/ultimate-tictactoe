"""Play 1000 games, write to pgn, calculate ordo stats"""
from tqdm import tqdm
from play import play
from players import *

result_strings = {
    1: '[Result "1-0"]\n1-0\n',
    2: '[Result "0-1"]\n0-1\n',
    3: '[Result "1/2-1/2"]\n1/2-1/2\n',
}


def make_stats(p1, p2, name1, name2, out_name, num=500, kt1=False, kt2=False):
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
    """
    with open(f'../ordo/{out_name}.pgn', '+w') as pgn:
        for _ in tqdm(range(num)):
            result, _ = play(p1, p2, verbose=False, press_enter=False, give_moves0=kt1, give_moves1=kt2)
            pgn.write(f'[White "{name1}"]\n')
            pgn.write(f'[Black "{name2}"]\n')
            pgn.write(result_strings[result])

            result, _ = play(p2, p1, verbose=False, press_enter=False, give_moves0=kt2, give_moves1=kt1)
            pgn.write(f'[White "{name2}"]\n')
            pgn.write(f'[Black "{name1}"]\n')
            pgn.write(result_strings[result])


if __name__ == '__main__':
    p0 = RandomPlayer()
    # make_stats(p0, p0, 'Random1', 'Random2', 'rand_rand', 5000)

    # p1 = TreePlayer(nodes=10)
    # make_stats(p0, p1, 'Random1', 'Tree10', 'rand_tree10', 500)

    # p1 = TreePlayer(nodes=100, v_mode=True)
    # p2 = TreePlayer(nodes=100, v_mode=False)
    # make_stats(p1, p2, 'Tree100V', 'Tree100N', 'tree100V_vs_N', 50)
    # make_stats(p1, p2, 'Tree100V', 'Tree100V_kt', 'keep_tree', 50, kt1=False, kt2=True)

    # make_stats(p0, p1, 'Random1', 'Tree100V', 'rand_tree100V', 50)
    # make_stats(p0, p1, 'Random1', 'Tree100V_kt', 'rand_tree100V_kt', 50, kt2=True)

    # Starting at 1000 nodes, all treeplayers will be kt
    p1 = TreePlayer(nodes=1000, v_mode=True)
    # p2 = TreePlayer(nodes=1000, v_mode=False)
    # make_stats(p1, p2, 'Tree1000V', 'Tree1000N', 'tree1000V_vs_N', 5, True, True)

    from gui import GuiPlayer
    p2 = GuiPlayer(x=600)
    make_stats(p1, p2, 'Tree1000V', 'Felix', 'tree1000V_felix', 2, kt1=True, kt2=True)

