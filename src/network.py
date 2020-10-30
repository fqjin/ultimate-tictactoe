import torch
import torch.nn as nn


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

        # 8 full 9x9 input planes:
        # - mover, X, O, legal moves, big empty tiled, big X, big O, big T
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

#######################
# Strided Architecture
#######################


class StridedBlock(nn.Module):
    """Uses stride to avoid conv overlapping boards"""
    def __init__(self, filters=16):
        super().__init__()
        self.blowup3 = nn.Upsample(scale_factor=3, mode='nearest')
        self.blowup9 = nn.Upsample(scale_factor=9, mode='nearest')
        self.relu = nn.ReLU(inplace=True)
        self.shuffle = nn.PixelShuffle(3)

        self.conv9 = nn.Sequential(
            nn.Conv2d(filters, 9 * filters, kernel_size=3, stride=3,
                      padding=0, bias=False, groups=filters),
            self.shuffle,
            nn.BatchNorm2d(filters),
            self.relu,
            nn.Conv1d(filters, filters, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(filters),
            self.relu,
        )

        self.conv9to3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=3,
                      padding=0, bias=False),
            nn.BatchNorm2d(filters),
            self.relu,
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * filters, filters, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(filters),
            self.relu,
            nn.Conv2d(filters, 9 * filters, kernel_size=3, stride=1,
                      padding=0, bias=False, groups=filters),
            self.shuffle,
            nn.BatchNorm2d(filters),
            self.relu,
            nn.Conv2d(filters, filters, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(filters),
            self.relu,
        )

        self.conv3to1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(filters),
            self.relu,
        )
        self.compress1 = nn.Conv2d(2*filters, filters, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x9, x3, x1):
        x9_ = self.conv9(x9)

        x3_ = self.conv9to3(x9)
        x3_ = torch.cat([x3, x3_], dim=1)
        x3_ = self.conv3(x3_)

        x1_ = self.conv3to1(x3)
        x1_ = torch.cat([x1, x1_], dim=1)
        x1_ = self.compress1(x1_)

        return x9 + x9_, x3 + x3_, x1 + x1_


class UTTTNetS(nn.Module):
    def __init__(self, blocks=5, filters=16):
        super().__init__()
        self.blocks = blocks
        self.filters = filters

        # 9x9 input planes:
        # - X, O, legal moves, corresponding tile
        # 3x3 input planes:
        # - big X, big O, big T, expanded mover
        # - two ways to input: as sector and as tiled

        self.in_conv9 = nn.Conv2d(4, filters - 4, 1, padding=0, bias=False)
        self.in_conv3 = nn.Conv2d(4, filters - 4, 1, padding=0, bias=False)

        trunk = [StridedBlock(filters) for _ in range(blocks)]
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
    for m in [UTTTNet(blocks=5, filters=64),   # 452353
              UTTTNetS(blocks=5, filters=16),  # _49905
              ]:
        params = sum(p.numel() for p in m.parameters())
        print(params)
