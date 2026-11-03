"""DACT 的自监督损失。所有光流/控制点都是 1/4 分辨率、1/4 像素单位；需要全分辨率时用 upsample_flow。"""
import torch
import torch.nn.functional as F

from models.traj.basis import cycle_residual
from models.traj.common import coords_grid, in_bounds, swap_halves, upsample_flow, warp_by_flow
from models.traj.reversal import invert_fixed_point


def charbonnier(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps)


@torch.no_grad()
def fb_occlusion(f_ab, f_ba, alpha1=0.01, alpha2=0.5):
    """前后向一致性遮挡掩码（1 = 遮挡/不一致）。公式同 ARFlow utils/warp_utils.py 的
    get_occu_mask_bidirection（MIT）：|F_ab + F_ba(p+F_ab)|^2 > alpha1(|F_ab|^2+|F_ba(.)|^2) + alpha2。
    alpha2 按全分辨率像素设定；在 1/4 分辨率上用 alpha2/16。"""
    f_ba_w = warp_by_flow(f_ba, f_ab, padding_mode="zeros")
    diff = f_ab + f_ba_w
    mag = (f_ab ** 2).sum(1, keepdim=True) + (f_ba_w ** 2).sum(1, keepdim=True)
    return ((diff ** 2).sum(1, keepdim=True) > alpha1 * mag + alpha2).to(f_ab.dtype)


def photometric_min_loss(im0, im1, gts, bwd_list):
    """主要的轨迹监督：两帧分别沿 F_{t->0}、F_{t->1} warp 到每个 GT 时刻，逐像素取误差较小的一边
    （时刻 t 的像素至少在一帧里可见，取最小值自动处理遮挡；思路同 Monodepth2 的 per-pixel minimum
    reprojection，只借鉴思路，该仓库许可证不允许商用）。
    gts: (B,T,3,H,W)；bwd_list: 长度 T 的列表，每项 (2B,2,h,w) = [F_{t->0}; F_{t->1}]。"""
    b, T, _, H, W = gts.shape
    src = torch.cat([im0, im1], 0)
    base = coords_grid(2 * b, H, W, im0.device, im0.dtype)
    loss = 0.0
    for i in range(T):
        f = upsample_flow(bwd_list[i], H // bwd_list[i].shape[-2])
        err = charbonnier(warp_by_flow(src, f) - gts[:, i].repeat(2, 1, 1, 1)).mean(1, keepdim=True)
        err = err + (1 - in_bounds(base + f, H, W))          # 采样点出界的一边不会被选中
        loss = loss + torch.minimum(err[:b], err[b:]).mean()
    return loss / T


def endpoint_photo_loss(im0, im1, ctrl):
    """两帧光流的光度损失：I_oth(p + P_K(p)) ≈ I_ref(p)，前后向不一致的像素（遮挡）不计。"""
    H = im0.shape[-2]
    end = upsample_flow(ctrl[:, -1], H // ctrl.shape[-2])    # (2B,2,H,W) = [F_{0->1}; F_{1->0}]
    ref, oth = torch.cat([im0, im1], 0), torch.cat([im1, im0], 0)
    m = 1 - fb_occlusion(end, swap_halves(end))
    err = charbonnier(warp_by_flow(oth, end) - ref).mean(1, keepdim=True)
    return (err * m).sum() / m.sum().clamp_min(1)


def cycle_loss(basis, ctrl, times=(0.0, 0.25, 0.5, 0.75)):
    """跨锚点循环一致：X1(X0(p,1), tau) ≈ X0(p, tau)，只在前后向一致的像素上计算。
    鲁棒函数用 UnFlow 的广义 Charbonnier (|r|^2 + eps^2)^0.45。"""
    s = torch.tensor(times, device=ctrl.device, dtype=ctrl.dtype)
    r, y = cycle_residual(basis, ctrl, s)                      # (2B,M,2,h,w)
    with torch.no_grad():
        end = ctrl[:, -1]
        m = (1 - fb_occlusion(end, swap_halves(end), alpha2=0.5 / 16)) * in_bounds(y, *ctrl.shape[-2:])
    rho = ((r ** 2).sum(2) + 1e-4) ** 0.45                     # (2B,M,h,w)
    return (rho * m).sum() / (m.sum() * len(times)).clamp_min(1)


def smooth2_loss(ctrl, im0, im1, alpha=10.0):
    """边缘感知二阶平滑，作用在全部控制点上。公式同 ARFlow losses/loss_blocks.py 的 smooth_grad_2nd（MIT）。"""
    n, K, _, h, w = ctrl.shape
    img = F.interpolate(torch.cat([im0, im1], 0), size=(h, w), mode="bilinear", align_corners=False)
    flo = ctrl.flatten(1, 2)
    wx = torch.exp(-alpha * (img[..., :, 1:] - img[..., :, :-1]).abs().mean(1, keepdim=True))
    wy = torch.exp(-alpha * (img[..., 1:, :] - img[..., :-1, :]).abs().mean(1, keepdim=True))
    dxx = flo[..., :, 2:] - 2 * flo[..., :, 1:-1] + flo[..., :, :-2]
    dyy = flo[..., 2:, :] - 2 * flo[..., 1:-1, :] + flo[..., :-2, :]
    return (wx[..., :, 1:] * dxx.abs()).mean() / 2 + (wy[..., 1:, :] * dyy.abs()).mean() / 2


def cm_timestamp_loss(basis, ctrl, voxel, bin_stride=4, fp_iters=4, den_thr=0.05):
    """事件对比度损失（voxel 切片近似）。把 bin b（时刻 tau_b）的事件沿 F_{tau_b->锚点} 推回锚点时刻，
    统计每个像素、每种极性的平均“到锚点的时间距离”，最小化其平方和 / 有事件的像素数。
    公式：Zhu et al. CVPR 2019 的平均时间戳损失 + Hagenaars et al. NeurIPS 2021 的像素数归一化；
    可对照 tudelft/taming_event_flow loss/flow.py 的 iwe_formatting / focus_loss（MIT）。
    voxel: 原始 voxel (B,NB,H,W)（不要归一化）；ctrl: (2B,K,2,h,w)。需要 CUDA（softsplat）。"""
    from models.softsplat import softsplat
    n, K, _, h, w = ctrl.shape
    b, nb, H, _ = voxel.shape
    f = H // h
    v = F.avg_pool2d(voxel, f) * (f * f)                        # 求和池化到 1/4
    ev = torch.stack([v.clamp_min(0), (-v).clamp_min(0)], 2)     # (B,NB,2,h,w) 正/负极性计数（近似）
    ev = torch.cat([ev, ev], 0)                                  # 两个锚点用同一批事件
    num = den = 0.0
    for k in range(0, nb, bin_stride):
        tau = k / (nb - 1)
        s = torch.cat([ctrl.new_full((b,), tau), ctrl.new_full((b,), 1 - tau)])   # 锚点 0 推到 0，锚点 1 推到 1
        back = invert_fixed_point(basis.displacement(ctrl, s), fp_iters)        # 时刻 tau_b 的像素 → 锚点
        e = ev[:, k]
        num = num + softsplat(e * s.view(-1, 1, 1, 1), back, None, "sum")
        den = den + softsplat(e, back, None, "sum")
    m = (den.detach() > den_thr).to(den.dtype)                  # 覆盖太少的像素不计，避免除小数的梯度尖峰
    ts = num / den.clamp_min(den_thr) * m
    nz = m.amax(1).flatten(1).sum(1).clamp_min(1)
    return ((ts ** 2).flatten(1).sum(1) / nz).mean()


class TrajLoss(torch.nn.Module):
    """汇总。端点/循环项对每次迭代的预测加权（RAFT sequence_loss 的 gamma^(N-1-i)，按权重和归一化），
    光度/平滑/CM 只用最后一次迭代。返回 (总损失, 各项数值字典)。"""

    def __init__(self, basis, w_ph=0.5, w_end=0.2, w_cyc=0.02, w_sm=0.05, w_cm=0.0, gamma=0.8):
        super().__init__()
        self.basis, self.gamma = basis, gamma
        self.w = {"ph": w_ph, "end": w_end, "cyc": w_cyc, "sm": w_sm, "cm": w_cm}

    def forward(self, preds, im0, im1, gts, bwd_list, voxel):
        n = len(preds)
        ws = [self.gamma ** (n - 1 - i) for i in range(n)]
        z = sum(ws)
        terms = {}
        if self.w["end"] > 0:
            terms["end"] = sum(wi * endpoint_photo_loss(im0, im1, p) for wi, p in zip(ws, preds)) / z
        if self.w["cyc"] > 0:
            terms["cyc"] = sum(wi * cycle_loss(self.basis, p) for wi, p in zip(ws, preds)) / z
        if self.w["sm"] > 0:
            terms["sm"] = smooth2_loss(preds[-1], im0, im1)
        if self.w["ph"] > 0:
            terms["ph"] = photometric_min_loss(im0, im1, gts, bwd_list)
        if self.w["cm"] > 0:
            terms["cm"] = cm_timestamp_loss(self.basis, preds[-1], voxel)
        total = sum(self.w[k] * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}
