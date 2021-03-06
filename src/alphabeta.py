import numpy as np
from tree import value_dict
from players import BasePlayer


class ABTree:
    """Alpha-beta minimax search tree"""
    def __init__(self,
                 board,
                 alpha,
                 beta,
                 **kwargs):
        self.board = board
        self.alpha = alpha
        self.beta = beta
        self.kwargs = kwargs
        self.sign = value_dict[self.board.mover + 1]
        self.v = None
        self.children = []

    def eval_fn(self):
        return value_dict[self.board.result]

    def update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def init_children(self):
        # Only init children when evaluated
        legal_moves = np.array(self.board.legal_moves)
        for sector, tile in zip(*legal_moves.nonzero()):
            board = self.board.copy()
            board.move(sector, tile)
            child = self.__class__(board, self.alpha, self.beta, **self.kwargs)
            child.v = child.eval_fn()
            self.children.append([sector, tile, child])

    def reorder_children(self):
        self.children.sort(key=lambda c: c[2].v, reverse=self.sign > 0)

    def explore(self, depth=1):
        if not depth or self.board.result:
            return
        if not self.children:
            self.init_children()
        self.reorder_children()
        value = -self.sign
        for child in self.children:
            child[2].update(self.alpha, self.beta)
            child[2].explore(depth=depth-1)
            value = self.sign*max(self.sign * value, self.sign * child[2].v)
            if self.sign > 0:
                self.alpha = max(self.alpha, value)
                if self.alpha >= self.beta:
                    break
            else:
                self.beta = min(self.beta, value)
                if self.beta <= self.alpha:
                    break
        self.v = value

    def child_values(self):
        return [c[2].v for c in self.children]

    def best_child(self):
        if not self.children:
            raise RuntimeError('children not initialized')
        if self.sign > 0:
            idx = np.argmax(self.child_values())
        else:
            idx = np.argmin(self.child_values())
        return self.children[idx.item()]

    def draw(self):
        self.board.draw()

    def goto(self, *args):
        child = self
        for i in args:
            child = child.children[i][2]
        return child


class RandABTree(ABTree):
    def eval_fn(self):
        if self.board.result:
            return value_dict[self.board.result]
        else:
            return 2*np.random.rand() - 1


class ABPlayer(BasePlayer):
    def __init__(self, treeclass=ABTree, max_depth=81, selfplay=False,
                 **kwargs):
        self.treeclass = treeclass
        self.max_depth = max_depth
        self.selfplay = selfplay
        self.kwargs = kwargs
        self.t = None

    def reset(self):
        self.t = None

    def get_move(self, board, moves=None, invtemp=None):
        if moves is None or self.t is None:
            self.t = self.treeclass(board, -1, 1, **self.kwargs)
        else:
            if self.selfplay:
                # Only want last move if selfplay
                moves = [moves[-1]]
            for m in moves:
                if not self.t.children:
                    self.t = self.treeclass(board, -1, 1, **self.kwargs)
                    break
                for c in self.t.children:
                    if c[0] == m[0] and c[1] == m[1]:
                        self.t = c[2]
                        break
                else:
                    raise RuntimeError('Given move not found in children')

        for d in range(1, self.max_depth+1):
            # Reset alpha beta
            self.t.alpha = -1
            self.t.beta = 1
            self.t.explore(d)

        if invtemp:
            noise = np.random.rand(len(self.t.children))
            noise -= 0.5
            noise /= invtemp
            for n, c in zip(noise, self.t.children):
                c[2].v += n

        return self.t.best_child()[:2]
