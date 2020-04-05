"""Game Tree Logic"""
import numpy as np
from typing import Union
from engine import BigBoard


ALPHA = np.full(81, 1.0)  # alpha = 10 / ave # legal moves
CPUCT = 2.0
value_dict = {
    1: 1.0,
    2: -1.0,
    3: 0.0,
}


class Root:
    def __init__(self):
        self.N = [1]
        self.Q = [0.0]
        self.terminal = [0]


class Tree:
    """Game Tree"""
    def __init__(self,
                 board: BigBoard,
                 parent: Union['Tree', Root],
                 parent_index=0):
        self.board = board
        self.parent = parent
        self.index = parent_index

        if self.board.result:
            self.parent.terminal[self.index] = 1
            self.parent.Q[self.index] = value_dict[self.board.result]
            return

        self.v = self.get_v()
        p_tmp = self.get_p()
        self.P = []
        self.children = []
        for sector, tile in np.ndindex(9, 9):
            if self.board.legal_moves[sector][tile]:
                # sector, tile, board
                # initialize child board on first call
                self.children.append([sector, tile])
                self.P.append(p_tmp[sector, tile])
        # TODO: Add logic for forcing moves

        self.P = np.asarray(self.P)
        # Start at N=1 to use P as prior
        # TODO: Actually compare N=0 vs N=1
        self.N = np.ones_like(self.P, dtype=np.int)
        self.Q = np.full_like(self.P, self.v)
        self.terminal = np.zeros_like(self.P, dtype=np.bool)

    def get_v(self):
        # v = value_head(self.board)
        v = self.board.states.count(1) - self.board.states.count(2)
        v /= 9
        return v

    def get_p(self):
        # p = policy_head(self.board)
        p = np.full((9, 9), 1/81)
        # Add Dirichlet noise
        noise = np.random.dirichlet(ALPHA).reshape((9, 9))
        p = 0.75 * p + 0.25 * noise
        # Mask and normalize
        p *= np.asarray(self.board.legal_moves)
        p /= p.sum()
        return p

    def check_terminal(self):
        if 0 not in self.terminal:
            terminal_v = np.min(self.Q) if self.board.mover else np.max(self.Q)
            self.parent.terminal[self.index] = 1
            self.parent.Q[self.index] = terminal_v

    def explore(self):
        puct = self.Q + CPUCT * self.P * np.sqrt(self.N.sum())/(1+self.N)
        puct -= puct.max() * self.terminal  # kill puct for terminal nodes so they are not visited
        puct_max = np.argmax(puct)
        return puct_max


