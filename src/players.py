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

    def get_move(self, board: BigBoard):
        # TODO: allow moves input so tree can be reused
        r = Root()
        t = Tree(board, r)
        for _ in range(self.nodes):
            t.explore()
            if r.terminal[0]:
                # print('Solved')
                break
        if self.v_mode:  # for low nodes
            index = np.argmax(t.sign * t.Q)
        else:
            index = np.argmax(t.N)
        return t.children[index][:2]
