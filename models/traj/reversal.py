"""轨迹反转：由锚点网格上的 F_{a->t} 得到时刻 t 网格上的 F_{t->a}。

- invert_fixed_point：不动点迭代 p <- x - D(p)（Chen et al., Med. Phys. 2008 的形变场求逆方法）。
  处处有定义、可微、不需要 CUDA；遮挡/折叠处只给出其中一个解。v1 默认用它。
- reverse_by_splat：用 softmax splatting（Niklaus & Liu, CVPR 2020；OF-AF 自带 models/softsplat.py，
  原作者声明仅限学术用途）把 -D 推到 p + D(p)，metric 大的像素胜出（前景）。会留下空洞，
  用另一锚点补（complement_from_other），再不行用不动点结果兜底。v2 用它。
"""
import torch

from .common import coords_grid, sample_at, swap_halves


def invert_fixed_point(disp, n_iter=4):
    """disp: (N,2,h,w) 锚点网格位移 D(p) → (N,2,h,w) 反向光流 R(x) ≈ -D(p*)，p* + D(p*) = x。
    收敛条件：位移场的局部雅可比范数 < 1（平滑运动下 4 次迭代误差 ~1e-3 像素）。"""
    n, _, h, w = disp.shape
    base = coords_grid(n, h, w, disp.device, disp.dtype)
    back = -disp
    for _ in range(n_iter):
        back = -sample_at(disp, base + back, padding_mode="border")
    return back


@torch.no_grad()
def splat_coverage(disp):
    """(N,1,h,w) 覆盖率：每个目标像素收到多少源像素（双线性权重之和）。不参与反传。"""
    from ..softsplat import softsplat          # 延迟导入：CPU 单元测试不需要 CuPy
    ones = disp.new_ones(disp.shape[0], 1, *disp.shape[2:])
    return softsplat(ones, disp, None, "sum")


def reverse_by_splat(disp, metric, cov_thr=0.5, den_min=0.02):
    """softmax splatting 反转。metric ∈ [-3,3]（estimator 已用 tanh 限幅），exp 后 ∈ [0.05,20]。
    返回 back (N,2,h,w) 与 valid (N,1,h,w，已 detach)。
    分母 clamp 到 den_min：覆盖率 >= cov_thr 的像素上分母 >= 0.025，不会出现 q/(q+1e-7) 式的梯度尖峰。"""
    from ..softsplat import softsplat
    wgt = metric.exp()
    out = softsplat(torch.cat([-disp * wgt, wgt], 1), disp, None, "sum")
    back = out[:, :2] / out[:, 2:3].clamp_min(den_min)
    valid = (splat_coverage(disp) >= cov_thr).to(disp.dtype)
    return back, valid


def complement_from_other(back, valid, end, fallback):
    """补空洞。back/valid/end/fallback 都是 (2B,...)，前 B 个属于锚点 0。
    锚点 0 在 x 处的空洞：先用锚点 1 的反向光流到 I1：y = x + F_{t->1}(x)，再走锚点 1 的端点位移到 I0：
        F_{t->0}(x) ≈ F_{t->1}(x) + F_{1->0}(y)
    另一锚点也无效时用 fallback（不动点结果）。"""
    n, _, h, w = back.shape
    back_o, valid_o, end_o = swap_halves(back), swap_halves(valid), swap_halves(end)
    comp = back_o + sample_at(end_o, coords_grid(n, h, w, back.device, back.dtype) + back_o,
                              padding_mode="border")
    return torch.where(valid > 0, back, torch.where(valid_o > 0, comp, fallback))
