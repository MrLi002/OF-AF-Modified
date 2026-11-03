"""特征编码器（全部输出 1/8 分辨率）。结构参考 RAFT core/extractor.py（BSD-3）的思路，
这里用 OF-AF 自带的 ResBlock，保持无归一化层（batch 常为 1，避免 BatchNorm）。"""
import torch
from torch import nn

from ..blocks import ResBlock


def _down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 2, 1), nn.LeakyReLU(0.2))


class MatchFeatEncoder(nn.Module):
    """匹配特征：OF-AF ImageEncoder 输出的 if_4 (N,128,H/4,W/4) → (N,C,H/8,W/8)。两帧共享权重。"""

    def __init__(self, in_ch=128, out_ch=128):
        super().__init__()
        self.net = nn.Sequential(_down(in_ch, 128), ResBlock(128, 128), nn.Conv2d(128, out_ch, 1))

    def forward(self, f4):
        return self.net(f4)


class EventSliceEncoder(nn.Module):
    """把 voxel 切成 S 个时间窗（每窗 win 个相邻 bin），共享 CNN 编码到 1/8。

    窗起点在 [0, NB-win] 上均匀分布，对时间反转对称（反转后第 k 窗 = 原来第 S-1-k 窗）。
    返回窗中心时间 times（局部时间，bin b 对应 b/(NB-1)，与 util/voxelization.py 一致）。
    """

    def __init__(self, voxel_bins=128, n_slices=9, win=16, out_ch=128):
        super().__init__()
        assert 2 <= n_slices and win <= voxel_bins
        self.win = win
        self.starts = [round(k * (voxel_bins - win) / (n_slices - 1)) for k in range(n_slices)]
        times = [(st + (win - 1) / 2) / (voxel_bins - 1) for st in self.starts]
        self.register_buffer("times", torch.tensor(times), persistent=False)
        self.net = nn.Sequential(_down(win, 64), _down(64, 96), _down(96, 128), ResBlock(128, out_ch))

    def forward(self, vox):
        """vox: (N,NB,H,W)（已归一化、局部时间）→ feats (N,S,C,H/8,W/8), times (S,)"""
        n = vox.shape[0]
        sl = torch.cat([vox[:, st:st + self.win] for st in self.starts], 0)   # (S*N,win,H,W)，窗优先
        f = self.net(sl)
        return f.view(len(self.starts), n, *f.shape[1:]).transpose(0, 1), self.times


class ContextEncoder(nn.Module):
    """GRU 的初始隐状态 net（tanh）与上下文 inp（relu），1/8 分辨率。
    输入：本锚点帧、另一帧、局部时间下的 voxel。"""

    def __init__(self, voxel_bins=128, hidden=128, ctx=128):
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(_down(6 + voxel_bins, 96), _down(96, 128), _down(128, 192),
                                 ResBlock(192, hidden + ctx))

    def forward(self, im_ref, im_oth, vox):
        x = self.net(torch.cat([im_ref, im_oth, vox], 1))
        net, inp = x.split([self.hidden, x.shape[1] - self.hidden], 1)
        return torch.tanh(net), torch.relu(inp)
