"""Game Tree Logic"""
import numpy as np
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
        self.N = [0]
        self.Q = [0.0]
        self.terminal = [0]


class Tree:
    """Game Tree"""
    def __init__(self, board: BigBoard, parent: 'Tree', parent_index):
        self.board = board
        self.parent = parent
        self.index = parent_index

        if self.board.result:
            self.parent.terminal[self.index] = 1
            self.parent.Q[self.index] = value_dict[self.board.result]
            return

        self.v = self.get_v()
        p = self.get_p()
        self.P = []
        self.children = []
        for sector, tile in np.ndindex(9, 9):
            if self.board.legal_moves[sector][tile]:
                # sector, tile, P, Q, board
                # initialize child board and Q on first call
                self.children.append((sector, tile))
                self.P.append(p[sector, tile])

        self.P = np.asarray(self.P)
        self.N = np.zeros_like(self.P)
        self.Q = np.full_like(self.P, self.v)
        self.terminal = np.zeros_like(self.P)

    def get_v(self):
        v = 0.0  # v = value_head(self.board)
        return v

    def get_p(self):
        p = np.full((9, 9), 1/81)  # p = policy_head(self.board)
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


