"""迭代更新模块。SepConvGRU / 运动编码器 / 凸上采样改写自 RAFT core/update.py、core/raft.py
（BSD-3-Clause），控制点输出头参考 uzh-rpg/bflow models/raft_spline/update.py 的 BezierHead（Apache-2.0）。"""
import torch
import torch.nn.functional as F
from torch import nn


class SepConvGRU(nn.Module):
    """与 RAFT 相同：先 1x5 再 5x1 的可分离卷积 GRU。"""

    def __init__(self, hidden_dim=128, input_dim=256):
        super().__init__()
        cin = hidden_dim + input_dim
        self.convz1 = nn.Conv2d(cin, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(cin, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(cin, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(cin, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(cin, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(cin, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        for cz, cr, cq in ((self.convz1, self.convr1, self.convq1), (self.convz2, self.convr2, self.convq2)):
            hx = torch.cat([h, x], 1)
            z, r = torch.sigmoid(cz(hx)), torch.sigmoid(cr(hx))
            q = torch.tanh(cq(torch.cat([r * h, x], 1)))
            h = (1 - z) * h + z * q
        return h


class TrajMotionEncoder(nn.Module):
    """RAFT BasicMotionEncoder 的扩展：除了相关特征和当前“光流”（这里是 2K 个控制点），
    还输入跨锚点循环残差（cyc）。输出 out_ch 通道（最后 2K 通道是控制点本身）。"""

    def __init__(self, corr_ch, K, cyc_ch, out_ch=128):
        super().__init__()
        self.convc1 = nn.Conv2d(corr_ch, 256, 1)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2 * K + cyc_ch, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(192 + 64, out_ch - 2 * K, 3, padding=1)

    def forward(self, ctrl_flat, cyc, corr):
        c = F.relu(self.convc2(F.relu(self.convc1(corr))))
        f = F.relu(self.convf2(F.relu(self.convf1(torch.cat([ctrl_flat, cyc], 1)))))
        return torch.cat([F.relu(self.conv(torch.cat([c, f], 1))), ctrl_flat], 1)


class TrajUpdateBlock(nn.Module):
    """一次迭代：运动编码 → GRU → 控制点增量 (2K) + 凸上采样掩码 (9*up*up)。"""

    def __init__(self, corr_ch, K, cyc_ch, hidden=128, ctx=128, up=2):
        super().__init__()
        self.encoder = TrajMotionEncoder(corr_ch, K, cyc_ch, 128)
        self.gru = SepConvGRU(hidden, ctx + 128)
        self.head = nn.Sequential(nn.Conv2d(hidden, 256, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 2 * K, 3, padding=1))
        self.mask = nn.Sequential(nn.Conv2d(hidden, 256, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 9 * up * up, 1))

    def forward(self, net, inp, corr, ctrl_flat, cyc):
        motion = self.encoder(ctrl_flat, cyc, corr)
        net = self.gru(net, torch.cat([inp, motion], 1))
        return net, self.head(net), 0.25 * self.mask(net)      # 0.25：RAFT 的梯度平衡系数


def convex_upsample(x, mask, r):
    """RAFT upsample_flow 的通用版本：x (N,C,h,w)、mask (N,9*r*r,h,w) → (N,C,r*h,r*w)，数值同时乘 r。
    控制点是像素单位的位移，所以和光流一样要乘 r；对 x 是线性的，先上采样控制点再求位移 = 先求位移再上采样。"""
    n, c, h, w = x.shape
    mask = mask.view(n, 1, 9, r, r, h, w).softmax(dim=2)
    up = F.unfold(r * x, [3, 3], padding=1).view(n, c, 9, 1, 1, h, w)
    up = (mask * up).sum(dim=2)                                    # (N,C,r,r,h,w)
    return up.permute(0, 1, 4, 2, 5, 3).reshape(n, c, r * h, r * w)
