"""Define UTTT player classes"""
import numpy as np
from random import randrange
from engine import BigBoard
from tree import Root, Tree


class BasePlayer:
    def get_move(self, board: BigBoard):
        raise NotImplementedError


class RandomPlayer(BasePlayer):
    def get_move(self, board: BigBoard):
        coords = np.nonzero(board.legal_moves)
        num_legal_moves = len(coords[0])
        move_index = randrange(num_legal_moves)
        return coords[0][move_index], coords[1][move_index]


class TreePlayer(BasePlayer):
    def __init__(self, nodes=0, v_mode=True):
        self.nodes = nodes
        self.v_mode = v_mode
        self.t = None

    def get_move(self, board: BigBoard, moves=None):
        r = Root()
        if moves is None:
            self.t = Tree(board, r)
        else:
            for m in moves:
                for c in self.t.children:
                    if c[0] == m[0] and c[1] == m[1]:
                        try:
                            self.t = c[2]
                            self.t.index = 0
                            self.t.parent = r
                        except IndexError:
                            self.t = Tree(board, r)
                        break
                else:
                    raise RuntimeError('Given move not found in children')

        # for _ in range(self.nodes - self.t.N.sum() + len(self.t.N)):
        for _ in range(self.nodes):
            self.t.explore()
            if r.terminal[0]:
                # print('Solved')
                break

        if self.v_mode:  # for low nodes
            index = np.argmax(self.t.sign * self.t.Q)
        else:
            index = np.argmax(self.t.N)
        return self.t.children[index][:2]
