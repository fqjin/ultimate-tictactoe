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
        self.v = 0
        self.N = [1]
        self.Q = [0.0]
        self.movesleft = [np.nan]

    def update(self):
        pass

    def increment(self):
        pass


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
        self.args = {'noise': noise}

        if self.board.result:
            self.v = value_dict[self.board.result]
            self.parent.movesleft[self.index] = 0
            self.parent.Q[self.index] = self.v  # N always = 1 in this case
            return
        # TODO: Add logic for forcing moves
        #  However, it seems only having 1 move is relatively rare
        #  But it may help with progagating deeper info up the tree

        self.init_part2()

    def init_part2(self):
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
        self.movesleft = np.full_like(self.P, np.nan)

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
        if not np.isnan(self.parent.movesleft[self.index]):
            q_over_n = self.Q / self.N
            if self.sign in q_over_n:
                # Winning terminal
                mask = self.sign != q_over_n
                puct_max = int(np.nanargmin(self.movesleft + 81*mask))
            elif 0 in q_over_n:
                # Drawn terminal
                mask = q_over_n != 0
                puct_max = int(np.nanargmin(self.movesleft + 81*mask))
            else:
                # Losing terminal
                puct_max = int(np.nanargmax(self.movesleft))
        else:
            puct = (self.sign*self.Q + CPUCT*self.P*np.sqrt(self.N.sum())) / self.N
            puct_max = int(np.argmax(puct))

        child = self.children[puct_max]
        if np.isnan(self.movesleft[puct_max]):  # Search child if not a known terminal node
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
        q_over_n = self.Q / self.N
        terminal = ~np.isnan(self.movesleft)
        if self.sign in q_over_n * terminal:
            # Win propagation
            self.v = self.sign
            self.parent.movesleft[self.index] = 1 + np.nanmin(
                self.movesleft[self.sign == q_over_n])
            self.parent.Q[self.index] = self.v * self.parent.N[self.index]
        elif np.all(terminal):
            # Draw/Loss propagation
            if self.board.mover:
                self.v = np.min(q_over_n)
            else:
                self.v = np.max(q_over_n)
            self.parent.movesleft[self.index] = 1 + np.nanmax(self.movesleft)
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
