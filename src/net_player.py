import torch
import numpy as np
from tree import Tree
from players import TreePlayer
from dataset import board_to_planes


class NetTree(Tree):
    def __init__(self, board, parent, parent_index=0, model=None, device='cpu'):
        assert model is not None
        self.device = device
        self.model = model  # use a single model.eval().cuda() outside tree init
        super().__init__(board, parent, parent_index)
        self.args = {'device': device, 'model': model}

    def get_p_and_v(self):
        x = board_to_planes(self.board)
        with torch.no_grad():
            p, v = self.model(x.to(self.device))
            p = p.exp_()
            p = torch.stack(torch.chunk(p, 3, dim=1), dim=1)
            p = torch.stack(torch.chunk(p, 3, dim=3), dim=2)
            p = p.view(9, 9).cpu().numpy()
            v = v.data.item()
        return self.add_dirichlet(p), v


class NetPlayer(TreePlayer):
    def __init__(self, nodes=0, v_mode=True, selfplay=False, model=None, device='cpu', savelist=None):
        assert model is not None
        super().__init__(nodes, v_mode, selfplay)
        self.treeclass = NetTree
        self.treeargs = {'model': model, 'device': device}
        self.savelist = savelist

    def get_move(self, board, moves=None):
        retvalue = super().get_move(board, moves)
        if self.savelist is not None:
            N_grid = np.zeros((9, 9))
            for c, n in zip(self.t.children, self.t.N):
                N_grid[c[0], c[1]] = n
                # TODO: Add terminal logic
                #  but not necessary if all positions get 1000 nodes (no early break)
            self.savelist.append(N_grid)
        return retvalue
