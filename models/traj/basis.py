"""轨迹基函数。参考：uzh-rpg/bflow models/raft_spline/bezier.py（BezierCurves._get_time_coeffs、
get_flow_from_reference），Apache-2.0。这里改成纯 torch、支持逐样本不同时间。"""
import math

import torch

from .common import coords_grid, sample_at, swap_halves


class BernsteinBasis:
    """D(s) = sum_{j=1..K} C(K,j) s^j (1-s)^(K-j) P_j

    s 是“从本锚点出发的局部时间”∈[0,1]：锚点 0 用 s = tau，锚点 1 用 s = 1 - tau。
    性质：D(0) = 0；D(1) = P_K（到另一帧的两帧光流）；P_j = (j/K) v 时 D(s) = s v（能精确表示匀速）。
    """

    def __init__(self, degree=4):
        self.K = degree
        self._binom = [float(math.comb(degree, j)) for j in range(1, degree + 1)]

    def weights(self, s):
        """s: 任意形状 (...) → (..., K)"""
        j = torch.arange(1, self.K + 1, device=s.device, dtype=s.dtype)
        c = torch.tensor(self._binom, device=s.device, dtype=s.dtype)
        s = s.unsqueeze(-1)
        return c * s.pow(j) * (1 - s).pow(self.K - j)

    def displacement(self, ctrl, s):
        """ctrl: (N,K,2,h,w)
        s: (N,) → 返回 (N,2,h,w)；s: (N,M) → 返回 (N,M,2,h,w)"""
        if s.dim() == 1:
            return torch.einsum("nk,nkchw->nchw", self.weights(s), ctrl)
        return torch.einsum("nmk,nkchw->nmchw", self.weights(s), ctrl)

    @staticmethod
    def endpoint(ctrl):
        """D(1) = P_K：锚点 0 为 F_{0->1}，锚点 1 为 F_{1->0}。"""
        return ctrl[:, -1]


def cycle_residual(basis, ctrl, s):
    """跨锚点循环残差（estimator 的输入特征和 cycle 损失共用）。
    ctrl: (2B,K,2,h,w)，前 B 个是锚点 0；s: (M,) 本锚点的局部时间。
    r(p,s) = [y + D_b(y, 1-s)] - [p + D_a(p, s)]，y = p + P_K(p) 是 p 在另一帧的位置。
    s=0 时就是前后向一致性误差；s=1 时恒为 0。返回 r (2B,M,2,h,w) 和 y (2B,2,h,w)。"""
    n, K, _, h, w = ctrl.shape
    end = ctrl[:, -1]
    y = coords_grid(n, h, w, ctrl.device, ctrl.dtype) + end
    other_y = sample_at(swap_halves(ctrl).flatten(1, 2), y, padding_mode="border").view(n, K, 2, h, w)
    s = s.to(ctrl.dtype).expand(n, -1)
    r = end.unsqueeze(1) + basis.displacement(other_y, 1 - s) - basis.displacement(ctrl, s)
    return r, y
