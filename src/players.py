"""Define UTTT player classes"""
import numpy as np
from random import randrange
from engine import BigBoard
from tree import Root, Tree


class BasePlayer:
    def get_move(self, board: BigBoard, moves=None, invtemp=None):
        raise NotImplementedError

    def resulted(self, board: BigBoard, moves):
        pass


class RandomPlayer(BasePlayer):
    def get_move(self, board: BigBoard, moves=None, invtemp=None):
        coords = np.nonzero(board.legal_moves)
        num_legal_moves = len(coords[0])
        move_index = randrange(num_legal_moves)
        return coords[0][move_index], coords[1][move_index]


class ManualPlayer(BasePlayer):
    """Manually enter sector and tile coordinates.

    To quit, input -1 for tile
    """
    def get_move(self, board: BigBoard, moves=None, invtemp=None):
        while True:
            if len(board.sectors) > 1:
                sector = int(input('Sector: '))
            else:
                sector = board.sectors[0]
                print(f'Sector {sector}')
            tile = int(input('Tile: '))
            if tile == -1:
                raise RuntimeError('QUIT')

            if board.legal_moves[sector][tile]:
                return sector, tile
            else:
                print('Illegal Move')


class TreePlayer(BasePlayer):
    def __init__(self, nodes=0, v_mode=True, selfplay=False):
        self.nodes = nodes
        self.v_mode = v_mode
        self.t = None
        self.treeclass = Tree
        self.treeargs = {}
        self.selfplay = selfplay

    def explore_fn(self):
        # Search nodes proportional to number of legal moves
        # is weaker than regular mode (100% CFS).
        # TODO: Need a smarter system like KLD_Gain.
        # nodes_left = self.nodes  # Allow nodes to accumulate
        nodes_left = self.nodes - self.t.N.sum() + len(self.t.N)
        for _ in range(nodes_left):
            self.t.explore()

    def get_move(self, board: BigBoard, moves=None, invtemp=None):
        r = Root()
        if moves is None:
            self.t = self.treeclass(board, r, **self.treeargs)
        else:
            try:
                if self.selfplay:
                    # Only want last move if selfplay
                    moves = [moves[-1]]
                for m in moves:
                    for c in self.t.children:
                        if c[0] == m[0] and c[1] == m[1]:
                            self.t = c[2]
                            self.t.index = 0
                            self.t.parent = r
                            break
                    else:
                        raise RuntimeError('Given move not found in children')
            except IndexError:
                self.t = self.treeclass(board, r, **self.treeargs)

        self.explore_fn()

        if self.v_mode:
            metric = self.t.sign * self.t.Q / self.t.N
        else:
            metric = self.t.N / self.t.N.max()

        if not invtemp:
            index = np.argmax(metric)
        else:
            if self.v_mode:
                metric -= metric.min()
            metric **= invtemp
            metric /= metric.sum()
            index = np.random.choice(range(len(metric)), p=metric)

        return self.t.children[index][:2]
