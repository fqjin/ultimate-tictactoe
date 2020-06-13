import torch
import numpy as np
from tree import Tree, Root, CPUCT
from players import TreePlayer
from dataset import board_to_planes


class NetTree(Tree):
    def __init__(self, board, parent, parent_index=0, model=None, device='cpu', noise=True):
        assert model is not None
        self.device = device
        self.model = model  # use a single model.eval().cuda() outside tree init
        super().__init__(board, parent, parent_index, noise)
        self.args = {'device': device, 'model': model, 'noise': noise}

    def get_p_and_v(self):
        x = board_to_planes(self.board)
        with torch.no_grad():
            p, v = self.model(x.to(self.device))
            p = p.exp_()
            p = torch.stack(torch.chunk(p, 3, dim=1), dim=1)
            p = torch.stack(torch.chunk(p, 3, dim=3), dim=2)
            p = p.view(9, 9).cpu().numpy()
            v = v.item()
        if self.noise:
            p = self.add_dirichlet(p)
        return p, v


class NetPlayer(TreePlayer):
    def __init__(self, nodes=0, v_mode=True, selfplay=False,
                 model=None, device='cpu', noise=True, savelist=None):
        assert model is not None
        super().__init__(nodes, v_mode, selfplay)
        self.treeclass = NetTree
        self.treeargs = {'model': model, 'device': device, 'noise': noise}
        self.savelist = savelist

    def get_move(self, board, moves=None, invtemp=None):
        retvalue = super().get_move(board, moves, invtemp)
        if self.savelist is not None:
            N_grid = np.zeros((9, 9))
            for c, n in zip(self.t.children, self.t.N):
                N_grid[c[0], c[1]] = n
                # TODO: Add terminal rescaling
                #  but not necessary if all positions get 1000 nodes (no early break)
            self.savelist.append(N_grid)
        return retvalue


class BatchNetPlayer(NetPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.treeclass = BatchNetTree

    def explore_fn(self):
        nodes_left = self.nodes - self.t.N.sum() + len(self.t.N)
        while nodes_left:
            nodes_searched = self.t.explore_batch(nodes_left=nodes_left)
            nodes_left -= nodes_searched


class BatchNetTree(NetTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.board.result:
            self.evaluated = True
        elif isinstance(self.parent, Root):
            super().init_part2()
            self.evaluated = True
        else:
            self.evaluated = False
            self.placeholder_v = self.parent.v

    def init_part2(self):
        pass

    def explore_batch(self, nodes_left):
        # Collect batch
        batch = []
        term_batch = []
        nonterm_batch = []
        for i in range(nodes_left):
            node = self.explore()
            if node in batch:
                break
            else:
                node.increment()
                batch.append(node)
                if node.parent.terminal[node.index]:
                    term_batch.append(node)
                else:
                    nonterm_batch.append(node)
        # print(f'Got {len(batch)} nodes in a batch, {len(term_batch)} terminal')
        for node in term_batch:
            # If node is terminal, its value and increment is correct on init.
            # Only parent needs to be updated, not self.
            node.parent.update()
        if nonterm_batch:
            # Batch predict
            x = torch.cat([board_to_planes(node.board) for node in nonterm_batch])
            with torch.no_grad():
                p, v = self.model(x.to(self.device))
                p = p.exp_()
                p = torch.stack(torch.chunk(p, 3, dim=1), dim=1)
                p = torch.stack(torch.chunk(p, 3, dim=3), dim=2)
                p = p.view(-1, 9, 9).cpu().numpy()
            # Distribute p and v and propagate
            for node, p_i, v_i in zip(nonterm_batch, p, v):
                node.set_pv(p_i, v_i.item())
                node.update()
        return len(batch)

    def explore(self):
        if not self.evaluated:
            return self

        if self.parent.terminal[self.index]:
            q_over_n = self.Q / self.N
            if self.sign in q_over_n * self.terminal:
                mask = self.sign != q_over_n
                puct_max = int(np.nanargmin(self.movesleft + 81*mask))
            else:
                puct_max = int(np.nanargmax(self.movesleft))
        else:
            puct = (self.sign*self.Q + CPUCT*self.P*np.sqrt(self.N.sum())) / self.N
            puct_max = int(np.argmax(puct))

        child = self.children[puct_max]
        if self.terminal[puct_max]:  # Child is terminal
            return child[2]
        elif len(child) == 2:  # Child not initialized
            board = self.board.copy()
            board.move(child[0], child[1])
            child.append(self.__class__(board,
                                        parent=self,
                                        parent_index=puct_max,
                                        **self.args))
            return child[2]
        else:
            return child[2].explore()

    def increment(self):
        """Increment to N and Q with placeholder value"""
        self.parent.N[self.index] += 1
        if not self.evaluated:
            self.parent.Q[self.index] += self.placeholder_v
        else:
            self.parent.Q[self.index] += self.v
        self.parent.increment()

    def set_pv(self, p_tmp, v):
        # TODO: This is essentially init_part2
        #  Try to merge this code into superclass?
        if self.noise:
            p_tmp = self.add_dirichlet(p_tmp)
        self.v = v
        self.P = []
        self.children = []
        for sector, tile in np.ndindex(9, 9):
            if self.board.legal_moves[sector][tile]:
                self.children.append([sector, tile])
                self.P.append(p_tmp[sector, tile])
        self.P = np.asarray(self.P)
        self.N = np.ones_like(self.P, dtype=np.int)
        self.Q = np.full_like(self.P, self.v)
        self.terminal = np.zeros_like(self.P, dtype=np.bool)
        self.movesleft = np.full_like(self.P, np.nan)

    def update(self):
        """Updates Q and v to correct values"""
        if not self.evaluated:  # Newly evaluated nodes
            self.evaluated = True
            self.parent.Q[self.index] -= self.placeholder_v
            self.parent.Q[self.index] += self.v
            self.parent.update()
        else:
            q_over_n = self.Q / self.N
            if self.sign in q_over_n * self.terminal:
                # Win
                self.v = self.sign
                self.parent.terminal[self.index] = True
                self.parent.movesleft[self.index] = 1 + np.nanmin(
                    self.movesleft[self.sign == q_over_n])
                self.parent.Q[self.index] = self.v * self.parent.N[self.index]
            elif 0 not in self.terminal:
                # Full terminal
                if self.board.mover:
                    self.v = np.min(self.Q / self.N)
                else:
                    self.v = np.max(self.Q / self.N)
                self.parent.terminal[self.index] = True
                self.parent.movesleft[self.index] = 1 + np.nanmax(self.movesleft)
                self.parent.Q[self.index] = self.v * self.parent.N[self.index]
            else:
                self.parent.Q[self.index] -= self.v
                self.v = self.Q.sum() / self.N.sum()
                self.parent.Q[self.index] += self.v
            self.parent.update()
