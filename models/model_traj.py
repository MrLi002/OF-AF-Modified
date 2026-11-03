"""DACT 版 OF-AF：保留 EventsEncoder / ImageEncoder / BiW，用 DualTrajEstimator + TrajQuery + TimeContext
替换 BiFEB（BidirFlow x2 + 16 步循环）和 Query。不改动 models/model.py。"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .BiW import BiW
from .encoder import EventsEncoder, ImageEncoder
from .traj import DualTrajEstimator, TimeContext, TrajQuery, event_feature_at


class DACTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        if not getattr(args, "fix_warp", False):
            raise ValueError("--flow_mode traj 需要同时打开 --fix_warp：轨迹模块和 BiW 必须用同一像素坐标约定")
        self.voxel_bins = args.voxel_bins
        self.events_encoder = EventsEncoder(args.voxel_bins, 32, bins_per_flow=args.voxel_bins // args.nb_of_flow)
        self.im_encoder = ImageEncoder(3, 32)
        self.traj = DualTrajEstimator(voxel_bins=args.voxel_bins, degree=args.traj_degree,
                                      iters=args.traj_iters, corr_impl=getattr(args, "corr_impl", "full"))
        self.traj_query = TrajQuery(self.traj.basis, reversal=args.traj_reversal)
        self.time_ctx = TimeContext()
        # fix_mask_grad 需要先打 ofaf_b1d3741_followup.patch；没打的话删掉这个参数
        self.biw = BiW(32, fix_warp=True, fix_mask_grad=getattr(args, "fix_mask_grad", False))

    def forward(self, im0, im1, event, interp_times, return_aux=False, return_traj=False, iters=None):
        """输入输出与 MyModel 相同；return_traj=True 时额外返回 dict(ctrl_preds, metric, flow_bwd) 给 TrajLoss。"""
        b = im0.shape[0]
        efs01 = self.events_encoder(im0, im1, event)               # (B,16,128,H/4,W/4)，只给 TimeContext 用
        if0_1, if0_4 = self.im_encoder(im0)
        if1_1, if1_4 = self.im_encoder(im1)
        preds, metric = self.traj(im0, im1, event, if0_4, if1_4, iters=iters)
        ctrl = preds[-1]
        f4 = torch.cat([if0_4, if1_4], 0)

        imst, auxs, bwd_list = [], [], []
        for i in range(interp_times.shape[1]):
            t = interp_times[:, i].to(im0.dtype)
            q = self.traj_query(ctrl, metric, t)
            fwd, bwd = q["flow_fwd"], q["flow_bwd"]
            ev_t = event_feature_at(efs01, t).repeat(2, 1, 1, 1)
            ctx = self.time_ctx(f4, bwd, ev_t, torch.cat([t, 1 - t]), q["valid"])
            args = (im0, im1, t.view(b, 1, 1, 1), fwd[:b], fwd[b:], bwd[:b], bwd[b:],
                    ctx[:b], ctx[b:], if0_1, if1_1)
            if self.training:
                out = checkpoint(self.biw, *args, use_reentrant=False, return_aux=return_aux)
            else:
                out = self.biw(*args, return_aux=return_aux)
            if return_aux:
                imst.append(out[0])
                auxs.append(out[1:])
            else:
                imst.append(out)
            bwd_list.append(bwd)

        result = torch.stack(imst, 1)
        if return_aux:     # 与 MyModel 相同：(preds, fwd0, fwd1, valid0, valid1)
            result = (result,) + tuple(torch.stack([a[k] for a in auxs], 1) for k in range(4))
        if return_traj:
            return result, {"ctrl_preds": preds, "metric": metric, "flow_bwd": bwd_list}
        return result
