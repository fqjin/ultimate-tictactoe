import torch
import torch.nn as nn


# class StrideBlock(nn.Module):
#     """Uses stride to avoid conv overlapping boards"""
#     def __init__(self, filters=64):
#         super().__init__()
#         self.compress = nn.Conv2d(filters, filters - filters//2 - filters//4,
#                                   kernel_size=1,
#                                   padding=0, bias=False)
#         # BN + ReLU
#         self.conv9 = nn.Conv2d(filters, filters//2, kernel_size=3, stride=3,
#                                padding=0, bias=False)
#         # BN + ReLU
#         self.conv3 = nn.Conv2d(filters//2, filters//4, kernel_size=3, stride=1,
#                                padding=0, bias=False)
#         # BN + ReLU
#         self.blowup3 = nn.Upsample(scale_factor=3, mode='nearest')
#         self.blowup9 = nn.Upsample(scale_factor=9, mode='nearest')
#
#     def forward(self, x9):
#         x3 = self.conv9(x9)  # Strided conv converts 9x9 to 3x3
#         x1 = self.conv3(x3)  # 3x3 conv converts 3x3 to 1x1
#         x = torch.cat([self.compress(x9), self.blowup3(x3), self.blowup9(x1)],
#                       dim=1)
#         return x


class SimpleBlock(nn.Module):
    def __init__(self, filters=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class UTTTNet(nn.Module):
    def __init__(self, blocks=5, filters=64):
        super().__init__()
        self.blocks = blocks
        self.filters = filters

        # Originally planning to do dual 9x9 and 3x3 residual paths,
        # but this seems a bit too complicated.
        # 8 full 9x9 input planes:
        # - mover, X, O, big empty tiled, big legal, big X, big O, big T
        # - empty 9x9 is NOT needed, see AlphaGoZero paper

        self.in_conv = nn.Conv2d(8, filters - 8, 3, padding=1, bias=False)

        trunk = [SimpleBlock(filters) for _ in range(blocks)]
        trunk.append(nn.BatchNorm2d(filters))
        trunk.append(nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*trunk)

        self.policy = nn.Sequential(
            nn.Conv2d(filters, filters, 1, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, 1, 1, padding=0, bias=False),
        )  # softmax

        self.value1 = nn.Sequential(
            nn.Conv2d(filters, filters, 3, stride=3, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )  # squeeze
        self.value2 = nn.Sequential(
            nn.Linear(filters, 1, bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        x = torch.cat([x, self.in_conv(x)], dim=1)
        x = self.trunk(x)
        p = self.policy(x)
        p = torch.log_softmax(p.view(-1, 81), dim=1).view(-1, 9, 9)
        v = self.value2(self.value1(x)[:, :, 0, 0])
        return p, v


if __name__ == '__main__':
    for m in [UTTTNet(blocks=5, filters=64),  # 452353
              ]:
        params = sum(p.numel() for p in m.parameters())
        print(params)
