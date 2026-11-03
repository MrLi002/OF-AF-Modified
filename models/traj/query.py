"""按查询时刻 t 生成 BiW 需要的输入：4 个光流 + 2 个上下文特征。替代 OF-AF 的 Query 和 ctxst 列表。"""
import math

import torch
from torch import nn

from ..blocks import ResBlock
from .common import warp_by_flow
from .reversal import complement_from_other, invert_fixed_point, reverse_by_splat


class TrajQuery(nn.Module):
    """ctrl (2B,K,2,h,w)、metric (2B,1,h,w)、t (B,) 物理时间 → dict：
        flow_fwd (2B,2,h,w)：[:B] = F_{0->t}，[B:] = F_{1->t}（锚点网格）
        flow_bwd (2B,2,h,w)：[:B] = F_{t->0}，[B:] = F_{t->1}（时刻 t 网格）
        valid    (2B,1,h,w)：反转结果是否来自本锚点（fixed_point 模式全 1）
    全部 1/4 分辨率、1/4 像素单位，直接对应 BiW 的 flow0t / flow1t / flowt0 / flowt1。"""

    def __init__(self, basis, reversal="fixed_point", fp_iters=4, cov_thr=0.5):
        super().__init__()
        assert reversal in ("fixed_point", "splat")
        self.basis, self.reversal, self.fp_iters, self.cov_thr = basis, reversal, fp_iters, cov_thr

    def forward(self, ctrl, metric, t):
        s = torch.cat([t, 1 - t], 0).to(ctrl.dtype)                    # 两个锚点各自的局部时间
        fwd = self.basis.displacement(ctrl, s)
        fp = invert_fixed_point(fwd, self.fp_iters)
        if self.reversal == "fixed_point":
            return {"flow_fwd": fwd, "flow_bwd": fp, "valid": torch.ones_like(fwd[:, :1])}
        spl, valid = reverse_by_splat(fwd, metric, self.cov_thr)
        bwd = complement_from_other(spl, valid, self.basis.endpoint(ctrl), fp)
        return {"flow_fwd": fwd, "flow_bwd": bwd, "valid": valid}


def event_feature_at(efs, t):
    """efs: (B,G,C,h,w) OF-AF EventsEncoder 的输出（G=16 组，组 g 大致对应时间 (g+0.5)/G）；
    t: (B,) → (B,C,h,w)，相邻两组线性插值。"""
    b, g = efs.shape[:2]
    pos = (t * g - 0.5).clamp(0, g - 1)
    lo = pos.floor().long().clamp(max=g - 2)
    wgt = (pos - lo.to(pos.dtype)).view(b, 1, 1, 1)
    idx = torch.arange(b, device=efs.device)
    return efs[idx, lo] * (1 - wgt) + efs[idx, lo + 1] * wgt


class TimeContext(nn.Module):
    """时刻 t 的上下文特征（替代 OF-AF 的 ctxt0/ctxt1，送进 BiW 的 RefBlock 和 refine 分支）。
    输入：对齐到 t 的锚点特征、t 附近的事件特征、反向光流、valid、局部时间的正余弦编码。"""

    def __init__(self, feat_ch=128, ev_ch=128, out_ch=128, n_freq=4):
        super().__init__()
        self.n_freq = n_freq
        cin = feat_ch + ev_ch + 2 + 1 + 2 * n_freq
        self.net = nn.Sequential(nn.Conv2d(cin, 128, 3, 1, 1), nn.LeakyReLU(0.2), ResBlock(128, out_ch))

    def embed(self, s):
        k = (2.0 ** torch.arange(self.n_freq, device=s.device, dtype=s.dtype)) * math.pi
        a = s[:, None] * k
        return torch.cat([a.sin(), a.cos()], 1)[..., None, None]

    def forward(self, f4, flow_bwd, ev_t, s, valid):
        """f4: (2B,128,h,w) = cat(if0_4, if1_4)；flow_bwd: (2B,2,h,w)；ev_t: (2B,C,h,w)；s: (2B,)"""
        n, _, h, w = f4.shape
        aligned = warp_by_flow(f4, flow_bwd)
        emb = self.embed(s.to(f4.dtype)).expand(-1, -1, h, w)
        return self.net(torch.cat([aligned, ev_t, flow_bwd / 8.0, valid, emb], 1))
