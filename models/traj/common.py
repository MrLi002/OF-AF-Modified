"""DACT 公共工具。坐标约定（全模块统一）：
- 光流/位移 (N,2,h,w)：通道 0 = x（水平，列方向），通道 1 = y（垂直，行方向），单位 = 所在分辨率的像素。
- 整数坐标 = 像素中心，采样用 align_corners=True（与 RAFT bilinear_sampler 相同，
  也与 OF-AF 打开 --fix_warp 后的 BackwardWarp 一致）。
"""
import torch
import torch.nn.functional as F


def coords_grid(n, h, w, device, dtype=torch.float32):
    """(n,2,h,w) 的像素坐标网格，[:,0]=x，[:,1]=y。"""
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                          torch.arange(w, device=device, dtype=dtype), indexing="ij")
    return torch.stack([x, y], 0)[None].repeat(n, 1, 1, 1)


def sample_at(field, coords, padding_mode="zeros"):
    """在绝对像素坐标 coords 处双线性采样 field。
    field: (N,C,H,W)；coords: (N,2,h,w) → (N,C,h,w)。"""
    H, W = field.shape[-2:]
    gx = 2.0 * coords[:, 0] / max(W - 1, 1) - 1.0
    gy = 2.0 * coords[:, 1] / max(H - 1, 1) - 1.0
    return F.grid_sample(field, torch.stack([gx, gy], -1), mode="bilinear",
                         padding_mode=padding_mode, align_corners=True)


def warp_by_flow(img, flow, padding_mode="border"):
    """后向 warp：out(x) = img(x + flow(x))。img 与 flow 分辨率相同。"""
    n, _, h, w = flow.shape
    return sample_at(img, coords_grid(n, h, w, flow.device, flow.dtype) + flow, padding_mode)


def in_bounds(coords, h, w):
    """(N,1,h',w')：坐标是否落在 [0,w-1]x[0,h-1] 内。"""
    x, y = coords[:, 0:1], coords[:, 1:2]
    return ((x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)).to(coords.dtype)


def upsample_flow(flow, factor=4):
    """与 BiW 内部相同的上采样方式：双线性放大，数值乘 factor。"""
    return F.interpolate(flow, scale_factor=factor, mode="bilinear") * factor


def normalize_voxel(v, eps=1e-6):
    """按样本除以非零元素的 RMS（保留符号，零仍为零）。只给编码器用；CM 损失用原始 voxel。"""
    nz = (v != 0).to(v.dtype)
    cnt = nz.flatten(1).sum(1).clamp_min(1)
    rms = ((v * v).flatten(1).sum(1) / cnt).sqrt()
    return v / (rms.view(-1, 1, 1, 1) + eps)


def swap_halves(x):
    """(2B,...) 的前后两半互换：锚点 0 的量 <-> 锚点 1 的量。"""
    b = x.shape[0] // 2
    return torch.cat([x[b:], x[:b]], 0)
