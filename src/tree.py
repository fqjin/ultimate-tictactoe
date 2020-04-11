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
                 parent_index=0,
                 noise=True):
        self.board = board
        self.sign = value_dict[self.board.mover + 1]  # (-1) ** self.board.mover
        self.parent = parent
        self.index = parent_index
        self.noise = noise
        self.args = {}

        if self.board.result:
            # print(f'Hit terminal node: {self.board.result}')
            self.v = value_dict[self.board.result]
            self.parent.terminal[self.index] = True
            self.parent.Q[self.index] = self.v  # N always = 1 in this case
            return

        # TODO: Add logic for forcing moves
        #  However, it seems only having 1 move is relatively rare
        #  But it may help with progagating deeper info up the tree
        # if np.sum(self.board.legal_moves):
        #     pass

        p_tmp, self.v = self.get_p_and_v()
        self.P = []
        self.children = []
        for sector, tile in np.ndindex(9, 9):
            if self.board.legal_moves[sector][tile]:
                # sector, tile, board
                # initialize child board on first call
                self.children.append([sector, tile])
                self.P.append(p_tmp[sector, tile])

        self.P = np.asarray(self.P)
        # Start at N=1 to use Q/P as prior
        # TODO: Actually compare N=0 vs N=1
        self.N = np.ones_like(self.P, dtype=np.int)
        self.Q = np.full_like(self.P, self.v)
        self.terminal = np.zeros_like(self.P, dtype=np.bool)

    def get_p_and_v(self):
        if self.noise:
            return self.add_dirichlet(self.get_p()), self.get_v()
        else:
            return self.get_p(), self.get_v()

    def get_v(self):
        v = self.board.states.count(1) - self.board.states.count(2)
        v /= 9
        return v

    def get_p(self):
        return np.full((9, 9), 1/81)

    def add_dirichlet(self, p):
        """Add Dirichlet noise"""
        noise = np.random.dirichlet(ALPHA).reshape((9, 9))
        p = 0.75 * p + 0.25 * noise
        # Mask and normalize
        p *= np.asarray(self.board.legal_moves)
        p /= p.sum()
        return p

    def explore(self):
        puct = (self.sign*self.Q + CPUCT*self.P*np.sqrt(self.N.sum())) / self.N
        puct_max = int(np.argmax(puct))

        child = self.children[puct_max]
        if not self.terminal[puct_max]:  # Not a known terminal node
            if len(child) == 2:  # Child not initialized
                board = self.board.copy()
                board.move(child[0], child[1])
                child.append(self.__class__(board,
                                            parent=self,
                                            parent_index=puct_max,
                                            **self.args))
            else:
                child[2].explore()

        self.N[puct_max] += 1
        self.Q[puct_max] += child[2].v

        # Check terminal
        if 0 not in self.terminal:
            # print('Full terminal')
            if self.board.mover:
                self.v = np.min(self.Q / self.N)
            else:
                self.v = np.max(self.Q / self.N)
            self.parent.terminal[self.index] = True
            self.parent.Q[self.index] = self.v * self.parent.N[self.index]
        elif self.sign in self.Q / self.N * self.terminal:
            # print('Win propagation')
            self.v = self.sign
            self.parent.terminal[self.index] = True
            self.parent.Q[self.index] = self.v * self.parent.N[self.index]
        else:
            self.v = self.Q.sum() / self.N.sum()

    def draw(self):
        self.board.draw()

    def goto(self, *args):
        child = self
        for i in args:
            child = child.children[i][2]
        return child

    def Q_over_N(self):
        return self.Q / self.N
