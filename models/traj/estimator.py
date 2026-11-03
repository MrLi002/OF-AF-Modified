"""双锚点连续轨迹估计器（DACT 的核心）。整体流程参考 uzh-rpg/bflow models/raft_spline/raft.py
（RAFTSpline.forward：多时刻轨迹索引查相关 + GRU 迭代更新控制点），Apache-2.0。

与 BFlow 的不同：
1. 两个锚点（I0、I1）各一组轨迹，拼成 2B 的 batch 共享权重；锚点 1 输入时间反转的事件。
2. 每次迭代把“跨锚点循环残差”作为额外输入，两组轨迹互相约束。
3. 没有轨迹真值，监督全部来自 losses/traj_losses.py 的自监督项和 BiW 的重建损失。
"""
import torch
import torch.nn.functional as F
from torch import nn

from .basis import BernsteinBasis, cycle_residual
from .common import coords_grid, in_bounds, normalize_voxel, swap_halves
from .corr import AltCorrBlock, CorrBlock
from .encoders import ContextEncoder, EventSliceEncoder, MatchFeatEncoder
from .update import TrajUpdateBlock, convex_upsample


class DualTrajEstimator(nn.Module):
    def __init__(self, voxel_bins=128, degree=4, n_slices=9, slice_win=16, target_slices=(2, 4, 6, 8),
                 cyc_times=(0.0, 0.25, 0.5, 0.75), corr_levels=4, corr_radius=3, iters=6,
                 feat_ch=128, hidden=128, ctx=128, detach_iter=True, corr_impl="full", metric_bound=3.0):
        super().__init__()
        assert all(0 < k < n_slices for k in target_slices)
        self.basis = BernsteinBasis(degree)
        self.K, self.iters, self.detach_iter, self.corr_impl = degree, iters, detach_iter, corr_impl
        self.levels, self.radius, self.metric_bound = corr_levels, corr_radius, metric_bound
        self.target_slices = list(target_slices)
        self.register_buffer("cyc_times", torch.tensor(cyc_times), persistent=False)

        self.match_enc = MatchFeatEncoder(128, feat_ch)
        self.slice_enc = EventSliceEncoder(voxel_bins, n_slices, slice_win, feat_ch)
        self.ctx_enc = ContextEncoder(voxel_bins, hidden, ctx)
        n_lookup = 1 + len(self.target_slices)                       # 1 个图像端点 + 若干事件时间窗
        corr_ch = n_lookup * corr_levels * (2 * corr_radius + 1) ** 2
        cyc_ch = 2 * len(cyc_times) + 1
        self.update = TrajUpdateBlock(corr_ch, degree, cyc_ch, hidden, ctx, up=2)
        self.metric_head = nn.Sequential(nn.Conv2d(hidden, 64, 3, 1, 1), nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 1, 3, 1, 1))

    def cycle_features(self, ctrl):
        """ctrl: (2B,K,2,h,w) → (2B, 2M+1, h, w)：tanh 压缩后的循环残差 + “y 是否在图内” 1 通道。"""
        r, y = cycle_residual(self.basis, ctrl, self.cyc_times)
        return torch.cat([torch.tanh(r.flatten(1, 2) / 8.0), in_bounds(y, *ctrl.shape[-2:])], 1)

    def forward(self, im0, im1, voxel, if0_4, if1_4, iters=None):
        """im0/im1: (B,3,H,W) ∈[0,1]，H、W 是 8 的倍数；voxel: (B,NB,H,W) 原始 voxel；
        if0_4/if1_4: (B,128,H/4,W/4)。
        返回 preds: list[(2B,K,2,H/4,W/4)]（训练时每次迭代一个，推理时只有最后一个；
        前 B 个是锚点 0，后 B 个是锚点 1；局部时间；1/4 像素单位）；
        metric: (2B,1,H/4,W/4) ∈[-b,b]，供 splat 反转选前景用。"""
        iters = iters or self.iters
        vox = normalize_voxel(voxel)
        vox_local = torch.cat([vox, -vox.flip(1)], 0)                  # 锚点 1：时间反转 + 极性取反
        im_ref, im_oth = torch.cat([im0, im1], 0), torch.cat([im1, im0], 0)

        fm = self.match_enc(torch.cat([if0_4, if1_4], 0))              # (2B,C,h8,w8)
        sl, s_times = self.slice_enc(vox_local)                        # (2B,S,C,h8,w8), (S,)
        Corr = CorrBlock if self.corr_impl == "full" else AltCorrBlock
        corr_fns = [Corr(fm, swap_halves(fm), self.levels, self.radius)]
        corr_fns += [Corr(sl[:, 0], sl[:, k], self.levels, self.radius) for k in self.target_slices]
        look_s = torch.stack([s_times.new_tensor(1.0)] + [s_times[k] for k in self.target_slices])

        net, inp = self.ctx_enc(im_ref, im_oth, vox_local)
        n, _, h, w = net.shape
        coords0 = coords_grid(n, h, w, net.device, net.dtype)
        ctrl = net.new_zeros(n, self.K, 2, h, w)
        look_s = look_s.to(net.dtype).expand(n, -1)                    # (2B,L)
        ref_s = s_times[:1].to(net.dtype).expand(n)                    # 参考时间窗的中心时间
        preds = []
        for it in range(iters):
            if self.detach_iter:
                ctrl = ctrl.detach()
            d = self.basis.displacement(ctrl, look_s)                  # (2B,L,2,h,w)
            d_ref = self.basis.displacement(ctrl, ref_s)               # (2B,2,h,w)
            coords = [coords0 + d[:, 0]]                               # 图像：I_ref(p) 对 I_oth(p + D(1))
            coords += [coords0 + d[:, i] - d_ref for i in range(1, d.shape[1])]   # 事件：相对参考窗的位移
            corr = torch.cat([f(c) for f, c in zip(corr_fns, coords)], 1)
            cyc = self.cycle_features(ctrl.detach())
            net, delta, up_mask = self.update(net, inp, corr, ctrl.flatten(1, 2), cyc)
            ctrl = ctrl + delta.view(n, self.K, 2, h, w)
            if self.training or it == iters - 1:
                up = convex_upsample(ctrl.flatten(1, 2), up_mask, 2)
                preds.append(up.view(n, self.K, 2, 2 * h, 2 * w))
        metric = F.interpolate(self.metric_head(net), scale_factor=2, mode="bilinear", align_corners=False)
        metric = self.metric_bound * torch.tanh(metric / self.metric_bound)
        return preds, metric
