"""相关体。CorrBlock 改写自 RAFT core/corr.py（BSD-3-Clause, princeton-vl）；
AltCorrBlock 是纯 PyTorch 的按需计算版本（RAFT 的 AlternateCorrBlock 依赖 CUDA 扩展 alt_cuda_corr，这里不用）。
两者输出数值相同（池化与点积可交换），通道排列也相同，训练用 full、推理大图可换 alt。"""
import math

import torch
import torch.nn.functional as F


def _window(r, device, dtype):
    d = torch.arange(-r, r + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    return torch.stack([dx, dy], -1).view(1, 1, (2 * r + 1) ** 2, 2)       # (1,1,N,2)，行优先


def _to_grid(c, H, W):
    return torch.stack([2 * c[..., 0] / max(W - 1, 1) - 1, 2 * c[..., 1] / max(H - 1, 1) - 1], -1)


class CorrBlock:
    """全对相关体 + 金字塔。fmap1/fmap2: (N,C,h,w) → __call__(coords (N,2,h,w)) → (N, L*(2r+1)^2, h, w)
    coords 是 fmap2 上的绝对像素坐标（本分辨率），即 “参考像素 + 位移”。"""

    def __init__(self, fmap1, fmap2, num_levels=4, radius=3):
        n, c, h, w = fmap1.shape
        corr = torch.einsum("nchw,ncuv->nhwuv", fmap1, fmap2) / math.sqrt(c)
        corr = corr.reshape(n * h * w, 1, h, w)
        self.pyramid = [corr]
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.pyramid.append(corr)
        self.radius, self.shape = radius, (n, h, w)

    def __call__(self, coords):
        n, h, w = self.shape
        delta = _window(self.radius, coords.device, coords.dtype)
        base = coords.permute(0, 2, 3, 1).reshape(n * h * w, 1, 1, 2)
        out = []
        for i, corr in enumerate(self.pyramid):
            c = base / 2 ** i + delta                                         # (Nhw,1,K,2)
            s = F.grid_sample(corr, _to_grid(c, *corr.shape[-2:]), align_corners=True)   # (Nhw,1,1,K)
            out.append(s.view(n, h, w, -1).permute(0, 3, 1, 2))
        return torch.cat(out, 1)


class AltCorrBlock:
    """不建全对体积，查询时对 fmap2 金字塔做局部采样再点积。显存 O(N*C*hw*(2r+1)^2)（单层），
    推理高分辨率时用；需要的话可对 hw 分块。"""

    def __init__(self, fmap1, fmap2, num_levels=4, radius=3):
        self.fmap1 = fmap1
        self.pyr2 = [fmap2]
        for _ in range(num_levels - 1):
            self.pyr2.append(F.avg_pool2d(self.pyr2[-1], 2, stride=2))
        self.radius = radius

    def __call__(self, coords):
        n, c, h, w = self.fmap1.shape
        delta = _window(self.radius, coords.device, coords.dtype)
        f1 = self.fmap1.flatten(2).transpose(1, 2).unsqueeze(2)               # (N,hw,1,C)
        base = coords.permute(0, 2, 3, 1).reshape(n, h * w, 1, 2)
        out = []
        for i, f2 in enumerate(self.pyr2):
            cc = base / 2 ** i + delta                                        # (N,hw,K,2)
            s = F.grid_sample(f2, _to_grid(cc, *f2.shape[-2:]), align_corners=True)   # (N,C,hw,K)
            corr = (s.permute(0, 2, 3, 1) * f1).sum(-1) / math.sqrt(c)        # (N,hw,K)
            out.append(corr.transpose(1, 2).reshape(n, -1, h, w))
        return torch.cat(out, 1)
