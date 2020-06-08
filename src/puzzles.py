import glob
import numpy as np
from engine import BigBoard
from gui import GuiPlayer


class Puzzle:
    def __init__(self, moves, index):
        self.moves = moves
        self.index = index
        self.game = BigBoard()
        for m in self.moves[:self.index]:
            self.game.move(*m)
        self.show()

    def show(self):
        p = GuiPlayer(x=600)
        while self.index < len(self.moves):
            sector, tile = p.get_move(self.game, moves=self.moves[:self.index])
            # TODO: Add smarter logic to calculate/allow alternative solutions
            # TODO: Related: Some win in x are not correct because network did
            #  play the best move (needs moves-left slope).
            # Suggestion: play against best_net starting from won position
            # Suggestion: try N_mode to favor best moves-left move
            if sector == self.moves[self.index][0] and tile == self.moves[self.index][1]:
                self.index += 2
                if self.index < len(self.moves):
                    self.game.move(*self.moves[self.index-2])
                    self.game.move(*self.moves[self.index-1])
                else:
                    self.game.move(*self.moves[self.index-2])
        p.resulted(self.game, self.moves)


class WinInPuzzle(Puzzle):
    def __init__(self, name, win_in, atol=1e-3):
        self.name = name
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
        super().__init__(moves, index)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--win_in', type=int, nargs='*', default=1,
                        help='Win in x moves. Default 1. nargs = *')
    parser.add_argument('--include', type=int, default=2000,
                        help='How many of the latest games to use.')
    args = parser.parse_args()

    if isinstance(args.win_in, int):
        args.win_in = [args.win_in]

    selfplay_games = sorted(glob.glob('../selfplay/*.npz'))
    selfplay_games = selfplay_games[-args.include:]
    np.random.shuffle(selfplay_games)

    for name in selfplay_games:
        WinInPuzzle(name, args.win_in)
