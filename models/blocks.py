import torch
import numpy as np
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
        )
        self.conv_block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x_ = self.conv_in(x)
        x_res = self.conv_block(x_)
        return x_res + x_


class ResBlockIF(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
        )
        self.conv_block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )
        self.conv_block1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x_ = self.conv_in(x)
        x_res = self.conv_block(x_) + x_
        x_res1 = self.conv_block1(x_res) + x_res
        return x_res1


class BackwardWarp(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = {}

    def forward(self, img, flow):
        N, _, H, W = img.size()

        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        key = f"{N}_{H}_{W}_{img.device}_{img.dtype}"
        if key not in self.grid:
            gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
            gridX = torch.tensor(gridX, requires_grad=False, device=img.device, dtype=img.dtype).unsqueeze(0).expand_as(u)
            gridY = torch.tensor(gridY, requires_grad=False, device=img.device, dtype=img.dtype).unsqueeze(0).expand_as(v)
            self.grid.update({
                key: [gridX, gridY]
            })
        gridX, gridY = self.grid[key]

        x = gridX + u
        y = gridY + v
        # range -1 to 1
        x = 2 * (x / W - 0.5)
        y = 2 * (y / H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=False)

        return imgOut


class DenseBlock(nn.Module):
    def __init__(self, ch_in):
        super(DenseBlock, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out