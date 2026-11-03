"""DACT 单元测试：python -m pytest -q tests/test_traj.py
前 8 个在 CPU 上就能跑；带 cuda 标记的需要 GPU + CuPy（softsplat）。"""
import types

import pytest
import torch

from models.traj.basis import BernsteinBasis, cycle_residual
from models.traj.common import coords_grid, sample_at
from models.traj.corr import AltCorrBlock, CorrBlock
from models.traj.estimator import DualTrajEstimator
from models.traj.reversal import invert_fixed_point
from models.traj.update import convex_upsample

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + CuPy")


def linear_ctrl(v, K, h, w):
    """匀速运动 D(s) = s v 对应的控制点 P_j = (j/K) v。v: (N,2)"""
    j = torch.arange(1, K + 1, dtype=v.dtype).view(1, K, 1, 1, 1) / K
    return (j * v.view(-1, 1, 2, 1, 1)).expand(-1, K, 2, h, w).contiguous()


def test_basis_endpoints_and_linear_precision():
    basis, K = BernsteinBasis(4), 4
    ctrl = torch.randn(2, K, 2, 5, 6)
    assert torch.allclose(basis.displacement(ctrl, torch.zeros(2)), torch.zeros(2, 2, 5, 6))
    assert torch.allclose(basis.displacement(ctrl, torch.ones(2)), ctrl[:, -1], atol=1e-6)
    v = torch.tensor([[3.0, -1.0], [0.5, 2.0]])
    lin = linear_ctrl(v, K, 5, 6)
    s = torch.tensor([0.3, 0.8])
    assert torch.allclose(basis.displacement(lin, s), s.view(2, 1, 1, 1) * v.view(2, 2, 1, 1), atol=1e-5)


def test_cycle_residual_zero_for_consistent_anchors():
    basis, K, h, w = BernsteinBasis(4), 4, 16, 20
    v = torch.tensor([[2.0, -1.0]])
    ctrl = torch.cat([linear_ctrl(v, K, h, w), linear_ctrl(-v, K, h, w)], 0)   # 锚点 1 局部时间里速度取反
    r, y = cycle_residual(basis, ctrl, torch.tensor([0.0, 0.25, 0.5, 0.75]))
    inner = (slice(None), slice(None), slice(None), slice(2, -2), slice(3, -3))
    assert r[inner].abs().max() < 1e-4


def test_fixed_point_inverse_affine():
    h, w = 40, 48
    base = coords_grid(1, h, w, "cpu")
    a = torch.tensor([[0.08, -0.05], [0.04, 0.10]])               # 小形变：雅可比范数 < 1
    disp = torch.einsum("ij,njhw->nihw", a, base - torch.tensor([w / 2, h / 2]).view(1, 2, 1, 1))
    disp = disp + torch.tensor([3.0, -2.0]).view(1, 2, 1, 1)
    back = invert_fixed_point(disp, n_iter=6)
    p = base + back
    land = p + sample_at(disp, p, padding_mode="border")
    inner = (slice(None), slice(None), slice(10, -10), slice(10, -10))
    assert (land - base)[inner].abs().max() < 1e-2


def test_convex_upsample_constant_interior():
    x = torch.full((1, 3, 6, 7), 2.0)
    up = convex_upsample(x, torch.randn(1, 9 * 4, 6, 7), 2)
    assert up.shape == (1, 3, 12, 14)
    assert torch.allclose(up[..., 2:-2, 2:-2], torch.full_like(up[..., 2:-2, 2:-2], 4.0), atol=1e-5)


def test_corr_full_equals_alt():
    f1, f2 = torch.randn(2, 16, 12, 16), torch.randn(2, 16, 12, 16)
    coords = coords_grid(2, 12, 16, "cpu") + torch.randn(2, 2, 12, 16) * 2
    a = CorrBlock(f1, f2, num_levels=3, radius=2)(coords)
    b = AltCorrBlock(f1, f2, num_levels=3, radius=2)(coords)
    assert a.shape == (2, 3 * 25, 12, 16)
    assert torch.allclose(a, b, atol=1e-4)


def test_corr_zero_motion_peak():
    f = torch.randn(1, 32, 10, 10)
    c = CorrBlock(f, f, num_levels=1, radius=1)(coords_grid(1, 10, 10, "cpu"))
    assert (c.argmax(1) == 4).float().mean() > 0.9                # 3x3 窗口中心 = 自身最相关


def test_estimator_shapes_and_time_reversal_symmetry():
    torch.manual_seed(0)
    est = DualTrajEstimator(voxel_bins=32, degree=3, n_slices=5, slice_win=8, target_slices=(2, 4),
                            corr_levels=2, corr_radius=2, iters=2, feat_ch=32, hidden=32, ctx=32).eval()
    B, H, W = 1, 64, 64
    im0, im1, vox = torch.rand(B, 3, H, W), torch.rand(B, 3, H, W), torch.randn(B, 32, H, W)
    f0, f1 = torch.randn(B, 128, H // 4, W // 4), torch.randn(B, 128, H // 4, W // 4)
    with torch.no_grad():
        p, m = est(im0, im1, vox, f0, f1)
        q, _ = est(im1, im0, -vox.flip(1), f1, f0)                  # 交换两帧 + 时间反转
    assert len(p) == 1 and p[-1].shape == (2 * B, 3, 2, H // 4, W // 4) and m.shape == (2 * B, 1, H // 4, W // 4)
    assert torch.allclose(p[-1][:B], q[-1][B:], atol=1e-4) and torch.allclose(p[-1][B:], q[-1][:B], atol=1e-4)


def test_estimator_gradients_reach_all_parts():
    """防止再出现 OF-AF 里“前向光流分支没有梯度”那一类问题。"""
    est = DualTrajEstimator(voxel_bins=32, degree=3, n_slices=5, slice_win=8, target_slices=(2, 4),
                            corr_levels=2, corr_radius=2, iters=2, feat_ch=32, hidden=32, ctx=32,
                            detach_iter=False).train()
    B, H, W = 1, 64, 64
    preds, metric = est(torch.rand(B, 3, H, W), torch.rand(B, 3, H, W), torch.randn(B, 32, H, W),
                        torch.randn(B, 128, 16, 16), torch.randn(B, 128, 16, 16))
    assert len(preds) == 2
    basis = est.basis
    loss = sum(basis.displacement(p, torch.full((2 * B,), 0.4)).abs().mean() for p in preds) + metric.mean()
    loss.backward()
    for name in ("match_enc", "slice_enc", "ctx_enc", "update", "metric_head"):
        g = sum(float(p.grad.abs().sum()) for p in getattr(est, name).parameters() if p.grad is not None)
        assert g > 0, name


@cuda
def test_splat_reversal_translation():
    from models.traj.reversal import reverse_by_splat
    disp = torch.zeros(1, 2, 32, 32, device="cuda")
    disp[:, 0], disp[:, 1] = 3.0, -2.0
    back, valid = reverse_by_splat(disp, torch.zeros(1, 1, 32, 32, device="cuda"))
    m = valid[0, 0].bool()
    assert m[4:-4, 4:-4].all()
    assert torch.allclose(back[0, 0][m], torch.full_like(back[0, 0][m], -3.0), atol=1e-4)
    assert torch.allclose(back[0, 1][m], torch.full_like(back[0, 1][m], 2.0), atol=1e-4)


@cuda
def test_full_model_forward_backward():
    from losses.traj_losses import TrajLoss
    from models.model_traj import DACTModel
    args = types.SimpleNamespace(voxel_bins=128, nb_of_flow=16, traj_degree=4, traj_iters=3,
                                 traj_reversal="splat", fix_warp=True, fix_mask_grad=True, corr_impl="full")
    model = DACTModel(args).cuda().train()
    B, T, H, W = 1, 3, 128, 128
    im0, im1 = torch.rand(B, 3, H, W, device="cuda"), torch.rand(B, 3, H, W, device="cuda")
    vox = torch.randn(B, 128, H, W, device="cuda")
    times = torch.tensor([[0.25, 0.5, 0.75]], device="cuda")
    out, traj = model(im0, im1, vox, times, return_aux=True, return_traj=True)
    assert out[0].shape == (B, T, 3, H, W)
    gts = torch.rand(B, T, 3, H, W, device="cuda")
    total, logs = TrajLoss(model.traj.basis, w_cm=0.1)(traj["ctrl_preds"], im0, im1, gts, traj["flow_bwd"], vox)
    (total + (out[0] - gts).abs().mean()).backward()
    assert all(torch.isfinite(torch.tensor(v)) for v in logs.values())
    norms = [p.grad.norm() for p in model.traj.parameters() if p.grad is not None]
    assert norms and torch.isfinite(torch.stack(norms)).all()
