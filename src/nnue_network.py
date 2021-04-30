import torch
import torch.nn as nn
# from torch.quantization import QuantStub, DeQuantStub
# Static quantized network was slower than float
#  perhaps because network size is too small
# However, torchscript trace reduced time by more than 50%


class NNUE(nn.Module):
    def __init__(self, f_in=360, f0=256, f1=32, f2=32, f_out=3):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.linear0 = nn.Linear(f_in, f0, bias=True)
        self.linear1 = nn.Linear(f0, f1, bias=True)
        self.linear2 = nn.Linear(f1, f2, bias=True)
        self.linear3 = nn.Linear(f2, f_out, bias=True)
        self.zeros = torch.zeros_like(self.linear0.weight)
        # self.relu0 = nn.ReLU(inplace=True)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        x = self.relu(self.linear0(x))
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        # x = self.dequant(x)
        return x

    def nnue_forward(self, f_new, f_old, x_old):
        """Forward pass using NNUE difference calculation"""
        # NNUE difference update is 3x slower than just forward pass
        # features must be batch size 1
        # features must be bool
        f_add = f_new & ~f_old
        f_sub = f_old & ~f_new
        x_add = torch.where(f_add, self.linear0.weight, self.zeros)
        x_sub = torch.where(f_sub, self.linear0.weight, self.zeros)
        x_new = x_old + torch.sum(x_add, dim=1) - torch.sum(x_sub, dim=1)
        x = self.relu(x_new)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x, x_new


if __name__ == '__main__':
    for m in [
        NNUE(f0=256, f1=32, f2=32),  # 101795
    ]:
        params = sum(p.numel() for p in m.parameters())
        print(params)
