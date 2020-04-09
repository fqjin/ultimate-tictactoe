import torch
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
            p = p.exp()
            p = p[0].cpu().numpy()
            v = v.data.item()
        return self.add_dirichlet(p), v


class NetPlayer(TreePlayer):
    def __init__(self, nodes=0, v_mode=True, selfplay=False, model=None, device='cpu', savelist=None):
        assert model is not None
        super().__init__(nodes, v_mode, selfplay)
        self.treeclass = NetTree
        self.treeargs = {'model': model, 'device': device}
