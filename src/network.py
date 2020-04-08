import torch
import torch.nn as nn


# Input planes:
# - empty 9x9 is NOT needed, see AlphaGoZero paper
# - X tile 9x9
# - O tile 9x9
# - legal sector 3x3
# - bigboard empty 3x3
#   - provide this tiled as 9x9
# - bigboard X 3x3
# - bigboard O 3x3
# - bigboard T 3x3
# - mover
# Consider 9 -> 3 -> 9 squeeze expand


class Block(nn.Module):
    def __init__(self, filters=64):
        super().__init__()
        self.BN1 = nn.BatchNorm2d()
        # RELU
        self.conv1 = nn.Conv2d()
        self.BN2 = nn.BatchNorm2d()
        # RELU
        self.conv2 = nn.Conv2d()
        # skip connect


# class ValueHead(nn.Module):
# class PolicyHead(nn.Module):


class UTTTNet(nn.Module):
    def __init__(self, blocks=5, filters=64):
        super().__init__()
        assert not filters % 2
        self.blocks = blocks
        self.filters = filters

        # Originally planning to do dual 9x9 and 3x3 residual paths,
        # but this seems a bit too complicated.

        # 9x9 inputs: X, O, mover
        self.in_conv9 = nn.Conv2d(3, filters//2, kernel_size=3, stride=3,
                                  padding=0, bias=False)

        # 3x3 inputs: legal sector, (empty) X, O, T, mover
        in_c3 = 4
        self.in_conv3 = nn.Sequential(
            nn.Conv2d(4+filters//2, 4+filters//2, kernel_size=3,
                      padding=0, bias=True, groups=4+filters//2),
            nn.Conv2d(4+filters//2, filters//2, kernel_size=1,
                      padding=0, bias=False),
        )

        # provide bigboard empty 3x3 tiled 9x9 on second pass
        self.blowup = nn.Upsample(scale_factor=3, mode='nearest')

    def forward(self, x9, x3, x_empty):
        in_x3 = torch.cat([x3, self.in_conv9(x9)])
        in_x3 = self.in_conv3(in_x3)

        in_x9 = torch.cat([x9, self.blowup(in_x3)])
        raise NotImplementedError
