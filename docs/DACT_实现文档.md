# 双锚点连续轨迹光流（DACT）实现文档

用来替换 OF-AF 中 BiFEB（BidirFlow×2 + 16 步循环）和 Flow Query 的新光流估计模块。DACT = Dual-Anchor Continuous Trajectory。

> **适用代码：** `MrLi002/OF-AF-Modified` 分支 `fix/ofaf-reproducibility-20261030` 的提交 b1d3741，并已应用 `ofaf_b1d3741_followup.patch`（要用到其中 BiW 的 `fix_mask_grad` 参数）。
>
> **当前状态：设计 + 参考实现，还没有跑过实验。**
>
> - 已验证：基函数的端点和匀速性质、不动点求逆的收敛性、循环残差和空洞补全公式、凸上采样的排布、全对相关与按需相关的等价性，都用 numpy 做了数值验证；合成数据生成器已实际运行，用轨迹真值 warp 的误差是 0.57/255。
> - 没验证：PyTorch 代码只做了语法检查和未定义名检查，**没有在 GPU 上运行过**。动手的第一步是跑 §9.1 的单元测试。
> - 损失权重、迭代次数等是起始值，需要按 §6.3 调。

---

## 0. 阅读和实现顺序

1. 先读 §1–§3：要解决什么、整体数据流、张量和坐标约定。最容易出错的两点是**坐标约定**和**局部时间**。
2. 按 §4 的文件表，从 `common.py` 写起。**每写完一个文件就跑对应的单元测试**（§9.1 的表里标了每个测试覆盖哪个文件）。
3. 然后是 §6 的损失、§7 的训练和评测脚本，最后按 §8 分阶段做实验。

---

## 1. 为什么要换：OF-AF 光流估计的缺陷

| # | OF-AF 的做法 | 问题 | DACT 的对应设计 |
| --- | --- | --- | --- |
| 1 | BidirFlow 按 1/16 的步长串行走 16 步，得到 17 个时间节点的光流，查询时分段线性插值 | 时间是离散的，节点之间速度不连续；不在节点上的时刻（如 GOPRO skip 6 的 k/7）只能插值；串行累积误差 | 每个像素一条连续轨迹，任意时刻闭式求值 |
| 2 | 前向 F₀→ₜ 与后向 Fₜ→₀ 由两个卷积头分别预测 | 两者没有几何一致性约束，可以互相矛盾 | Fₜ→₀ 由 F₀→ₜ 的轨迹反转得到，几何上自洽 |
| 3 | 两个方向（bidir_flow0 / bidir_flow1）权重独立、互不通信 | 同一个物理运动的两种描述可以不一致 | 两个锚点共享权重（锚点 1 输入时间反转的事件）；每次迭代输入跨锚点循环残差；损失里再加循环一致 |
| 4 | 前向光流只通过 avg splat 掩码那一条路径拿到梯度 | 原版几乎没有梯度（已用 `--fwd_loss_weight`、`--fix_mask_grad` 缓解） | 轨迹在每个 GT 中间时刻直接受光度损失监督 |
| 5 | 事件只作为卷积特征 | 没用上“同一条边缘产生的事件沿运动轨迹对齐”这一物理约束 | 事件对比度（CM）损失，第 3 阶段加入 |

---

## 2. 方法总览

### 2.1 轨迹表示

对锚点帧 a ∈ {0, 1} 的每个像素 p，用 K 个控制点 P₁…P_K（每个是一个 2 维位移）表示它在整个区间内的轨迹：

```
X_a(p, s) = p + D_a(p, s)
D_a(p, s) = Σ_{j=1..K} C(K, j) · s^j · (1 − s)^(K − j) · P_{a,j}(p)
```

- **s 是局部时间**：锚点 0 用 s = τ，锚点 1 用 s = 1 − τ（τ 是物理时间，I0 在 τ = 0，I1 在 τ = 1）。这样两个锚点用同一个公式、同一个网络。
- D(0) = 0，D(1) = P_K。所以 P_{0,K} 就是 F₀→₁，P_{1,K} 就是 F₁→₀。
- 匀速运动可以精确表示：P_j = (j/K)·v 时 D(s) = s·v。K = 4 能表示加速、减速和一次转弯。
- 这就是 BFlow 用的 Bézier 表示。BFlow 用它做事件光流，有轨迹真值监督；这里改成双锚点、自监督，并接到插帧上。

查询时刻 τ 时，BiW 需要的 4 个光流这样得到：

| BiW 参数 | 含义 | 所在网格 | 计算方法 |
| --- | --- | --- | --- |
| `flow0t` | F₀→τ | I0 的像素 | D₀(p, τ)，闭式 |
| `flow1t` | F₁→τ | I1 的像素 | D₁(q, 1 − τ)，闭式 |
| `flowt0` | Fτ→₀ | 时刻 τ 的像素 | 反转 D₀(·, τ)：v1 用不动点求逆；v2 用 softmax splat，空洞用锚点 1 补 |
| `flowt1` | Fτ→₁ | 时刻 τ 的像素 | 同上，两个锚点角色互换 |

### 2.2 数据流

```
im0, im1 (B,3,H,W)                          voxel (B,128,H,W)
   │                                             │
   ├─ ImageEncoder（保留）── if0_1 / if1_1 (B,32,H,W) ───────────────────────────┐
   │                     └── if0_4 / if1_4 (B,128,H/4,W/4) ──┐                    │
   ├─ EventsEncoder（保留）── efs01 (B,16,128,H/4,W/4) ───────┼──────────┐         │
   │                                                          ▼          │         │
   │   DualTrajEstimator（新，替代 BidirFlow×2 + 16 步循环）             │         │
   │     2B 的 batch：[锚点0: (I0, I1, V) ; 锚点1: (I1, I0, −flip(V))]   │         │
   │     匹配特征 / 事件时间窗特征 / 上下文 → 相关体（1/8 分辨率）       │         │
   │     → GRU 迭代 N 次更新控制点 → 凸上采样到 1/4                       │         │
   │     输出 ctrl (2B,K,2,H/4,W/4)、metric (2B,1,H/4,W/4)                │         │
   │                                                          │          │         │
   │   对每个查询时刻 τ：                                      ▼          ▼         │
   │     TrajQuery（新，替代 Query）     → flow0t, flow1t, flowt0, flowt1            │
   │     TimeContext（新，替代 ctxst 列表）→ ctxt0, ctxt1 (B,128,H/4,W/4)            │
   │                                                          │                    │
   └─────────────────────────────────────► BiW（保留，不改）◄─┴────────────────────┘
                                                  │
                                           预测帧 (B,3,H,W)

训练时：TrajLoss（新）用 ctrl、flowt0/flowt1、GT 中间帧、原始 voxel 计算自监督损失，和原来的重建损失相加。
```

### 2.3 与 OF-AF 各模块的对应

| OF-AF 模块 | 在 DACT 中 | 说明 |
| --- | --- | --- |
| `EventsEncoder` | 保留 | 只给 TimeContext 提供 τ 附近的事件特征 |
| `ImageEncoder` | 保留 | `if_1` 给 BiW 的 refine；`if_4` 给 MatchFeatEncoder 和 TimeContext |
| `BidirFlow` ×2 + 16 步循环 | 删除 | 由 DualTrajEstimator 替代 |
| `Query` | 删除 | 由 TrajQuery 替代 |
| `ctxst0` / `ctxst1` 列表 | 删除 | 由 TimeContext 按 τ 现算 |
| `BiW` | 保留，不改 | 输入的 6 个光流/特征张量，形状和含义与原来完全相同 |
| `softsplat`、`BackwardWarp` | 复用 | 轨迹模块用 align_corners=True 约定，所以必须开 `--fix_warp` |

**不改 `models/model.py`。** 新模型写在 `models/model_traj.py`，训练脚本用 `--flow_mode traj` 切换。原模型和官方权重的评测完全不受影响。

---

## 3. 符号、张量和坐标约定

| 符号 | 形状 | 含义 / 单位 |
| --- | --- | --- |
| B | – | batch 大小（训练时通常是 1） |
| H, W | – | 输入分辨率。DACT 要求是 **8 的倍数**（相关体在 1/8 分辨率） |
| h8, w8 / h4, w4 | – | H/8, W/8 / H/4, W/4 |
| NB | – | voxel 的 bin 数，默认 128。bin b 对应时间 b/(NB−1)（`util/voxelization.py` 按这个做三线性插值） |
| T | – | 每个样本的查询时刻数（GOPRO 训练是 31） |
| K | – | 控制点个数，即 Bernstein 次数，默认 4 |
| `im0`, `im1` | (B,3,H,W) | RGB，取值 [0,1] |
| `voxel` | (B,NB,H,W) | 原始 voxel：带符号的极性累加，未归一化 |
| `interp_times` | (B,T) | 物理时间 τ ∈ (0,1) |
| `ctrl` | (2B,K,2,h4,w4) | 控制点。前 B 个是锚点 0，后 B 个是锚点 1；按局部时间；单位是 1/4 分辨率的像素 |
| `metric` | (2B,1,h4,w4) | softmax splat 的前景度量，tanh 限幅到 [−3, 3] |
| 光流 / 位移 | (N,2,h,w) | 通道 0 = x（水平），通道 1 = y（垂直）；单位是所在分辨率的像素 |

**坐标约定（全模块统一，最容易出错）：**

1. 整数坐标就是像素中心。采样一律用 `grid_sample(..., align_corners=True)`，归一化为 `2x/(W−1) − 1`。这和 RAFT 的 `bilinear_sampler` 相同，也和 OF-AF 打开 `--fix_warp` 后的 `BackwardWarp` 一致。
2. 后向 warp：`out(x) = img(x + flow(x))`。softsplat：把源像素 p 推到 `p + flow(p)`。
3. 分辨率换算：1/8 → 1/4 用凸上采样，数值 ×2；1/4 → 全分辨率用双线性插值，数值 ×4（和 BiW 内部的做法相同）。
4. 2B batch 的前后两半互换就是两个锚点互换（`swap_halves`）。锚点 1 的 voxel 是 `-voxel.flip(1)`：时间反转时，亮度变亮变成变暗，极性也要反号。

---

## 4. 文件结构与职责

只新建文件，不动原有模型文件；需要修改的只有 `train_model.py` 和 `bsergb_eval.py`（见 §7）。

```
models/
  model_traj.py          DACTModel：组装各模块（对外接口和 MyModel 相同）
  traj/
    __init__.py
    common.py            坐标网格、采样、warp、voxel 归一化、swap_halves
    basis.py             BernsteinBasis（轨迹求值）、cycle_residual（跨锚点循环残差）
    encoders.py          MatchFeatEncoder、EventSliceEncoder、ContextEncoder
    corr.py              CorrBlock（全对相关）、AltCorrBlock（按需相关）
    update.py            SepConvGRU、TrajMotionEncoder、TrajUpdateBlock、convex_upsample
    estimator.py         DualTrajEstimator（核心：迭代估计两组控制点）
    reversal.py          invert_fixed_point、reverse_by_splat、complement_from_other
    query.py             TrajQuery、TimeContext、event_feature_at
losses/
  traj_losses.py         TrajLoss 和各个损失项
tests/
  test_traj.py           单元测试
tools/
  make_synthetic_traj.py 合成数据生成（带轨迹真值）
train_traj.sh            训练脚本示例
```

| 文件 | 类 / 函数 | 负责什么 | 输入 → 输出 | 参考（许可证） |
| --- | --- | --- | --- | --- |
| `common.py` | `coords_grid`、`sample_at`、`warp_by_flow`、`in_bounds`、`upsample_flow`、`normalize_voxel`、`swap_halves` | 按统一坐标约定的基础操作 | 见 §5.1 | RAFT `core/utils/utils.py` 的 `bilinear_sampler`、`coords_grid`（BSD-3） |
| `basis.py` | `BernsteinBasis` | 控制点 → 任意局部时间的位移 | ctrl (N,K,2,h,w)，s (N,) 或 (N,M) → (N,2,h,w) 或 (N,M,2,h,w) | bflow `models/raft_spline/bezier.py`（Apache-2.0） |
| `basis.py` | `cycle_residual` | 两个锚点的轨迹是否描述同一个运动 | ctrl (2B,K,2,h,w)，s (M,) → r (2B,M,2,h,w)，y (2B,2,h,w) | 本设计 |
| `encoders.py` | `MatchFeatEncoder` | 两帧的匹配特征 | (N,128,h4,w4) → (N,C,h8,w8) | RAFT `core/extractor.py` 的思路 |
| `encoders.py` | `EventSliceEncoder` | 事件时间窗特征 | (N,NB,H,W) → (N,S,C,h8,w8)，times (S,) | bflow `raft.py` 的 `gen_voxel_grids` |
| `encoders.py` | `ContextEncoder` | GRU 的初始隐状态和上下文 | im_ref、im_oth、vox → net、inp，各 (N,128,h8,w8) | RAFT 的 cnet |
| `corr.py` | `CorrBlock` / `AltCorrBlock` | 相关体和局部查表 | fmap1、fmap2 (N,C,h8,w8)，coords (N,2,h8,w8) → (N,L·(2r+1)²,h8,w8) | RAFT `core/corr.py`（BSD-3） |
| `update.py` | `TrajUpdateBlock` | 一次迭代：控制点增量 + 上采样掩码 | net、inp、corr、ctrl、cyc → net、Δctrl (N,2K,h8,w8)、mask (N,36,h8,w8) | RAFT `core/update.py`；bflow `update.py` |
| `update.py` | `convex_upsample` | 1/8 → 1/4 | (N,C,h,w)、mask → (N,C,2h,2w) | RAFT `raft.py` 的 `upsample_flow` |
| `estimator.py` | `DualTrajEstimator` | 估计两组控制点 | im0、im1、voxel、if0_4、if1_4 → preds（list，每项 (2B,K,2,h4,w4)）、metric | bflow `raft.py` 的 `RAFTSpline.forward` |
| `reversal.py` | `invert_fixed_point` | F_{a→t} → F_{t→a}，处处有定义 | (N,2,h,w) → (N,2,h,w) | Chen et al. 2008 |
| `reversal.py` | `reverse_by_splat`、`complement_from_other` | 带遮挡处理的反转 + 补空洞 | 见 §5.7 | softmax splatting（OF-AF 自带，仅限学术用途） |
| `query.py` | `TrajQuery` | 给定 τ，输出 BiW 的 4 个光流 | ctrl、metric、t (B,) → flow_fwd、flow_bwd、valid | 本设计 |
| `query.py` | `TimeContext`、`event_feature_at` | 给定 τ，输出 BiW 的 2 个上下文特征 | → (2B,128,h4,w4) | 本设计 |
| `model_traj.py` | `DACTModel` | 组装；对外接口和 MyModel 相同 | 同 MyModel，另加 `return_traj` | OF-AF `models/model.py` |
| `traj_losses.py` | `TrajLoss` 等 | 自监督损失 | 见 §6 | ARFlow（MIT）、taming_event_flow（MIT）；UnFlow、Monodepth2 只借鉴思路 |

`models/traj/__init__.py` 只负责导出：

`models/traj/__init__.py`：

```python
"""DACT：双锚点连续轨迹光流（Dual-Anchor Continuous Trajectory）。"""
from .basis import BernsteinBasis, cycle_residual
from .estimator import DualTrajEstimator
from .query import TimeContext, TrajQuery, event_feature_at
```

`losses/` 是新目录，如果你的环境里 `import losses.traj_losses` 报找不到，在里面放一个空的 `__init__.py`（`models/` 目录现在没有 `__init__.py`，靠的是 Python 的命名空间包，`losses/` 同理）。

---
## 5. 模块详解

每个模块按“负责什么 → 输入输出 → 参考 → 代码 → 要点”的顺序写。代码可以直接作为起点，测试见 §9.1。

### 5.1 `models/traj/common.py`：公共工具

**负责：** 统一的坐标约定。其他所有文件只通过这里的函数做采样、warp 和网格，不要在别处自己写 `grid_sample`。

| 函数 | 输入 | 输出 |
| --- | --- | --- |
| `coords_grid(n, h, w, device)` | – | (n,2,h,w)，`[:,0]` = x，`[:,1]` = y |
| `sample_at(field, coords)` | field (N,C,H,W)，coords (N,2,h,w) 绝对像素坐标 | (N,C,h,w) |
| `warp_by_flow(img, flow)` | 同分辨率的 img 和 flow | `img(x + flow(x))` |
| `in_bounds(coords, h, w)` | coords (N,2,·,·) | (N,1,·,·)，在图内为 1 |
| `upsample_flow(flow, f)` | (N,2,h,w) | (N,2,fh,fw)，数值 ×f |
| `normalize_voxel(v)` | (B,NB,H,W) | 除以非零元素的 RMS |
| `swap_halves(x)` | (2B,…) | 前后两半互换 |

**参考：** RAFT `core/utils/utils.py` 的 `bilinear_sampler`、`coords_grid`（BSD-3-Clause）。

`models/traj/common.py`：

```python
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
```

**要点：**

- `normalize_voxel` 只给编码器用，CM 损失必须用原始 voxel（它要的是事件计数）。
- `upsample_flow` 用 `F.interpolate` 的默认 `align_corners=False`，和 BiW 内部的上采样方式一致，这样损失里看到的全分辨率光流就是 BiW 实际用的那个。

### 5.2 `models/traj/basis.py`：轨迹基函数和循环残差

**负责：** 控制点 → 任意时刻的位移；计算两个锚点之间的循环残差。

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `BernsteinBasis(degree=K)` | – | – |
| `.weights(s)` | s 任意形状 | (…,K) |
| `.displacement(ctrl, s)` | ctrl (N,K,2,h,w)；s (N,) | (N,2,h,w) |
| 同上 | s (N,M) | (N,M,2,h,w) |
| `.endpoint(ctrl)` | ctrl | `ctrl[:, -1]`，即 D(1) |
| `cycle_residual(basis, ctrl, s)` | ctrl (2B,K,2,h,w)；s (M,) | r (2B,M,2,h,w)，y (2B,2,h,w) |

**参考：** bflow `models/raft_spline/bezier.py` 的 `BezierCurves._get_time_coeffs` 和 `get_flow_from_reference`（Apache-2.0）。BFlow 用 numpy + numba 计算系数、所有样本共用一组时间；这里改成纯 torch，每个样本可以有不同的时间。

`models/traj/basis.py`：

```python
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
```

**要点：**

- **局部时间：** 锚点 0 传 s = τ，锚点 1 传 s = 1 − τ。这一步只在 `TrajQuery` 和损失里做，估计器内部全部用局部时间。
- `(1 − s).pow(K − j)` 在 s = 1、j = K 时是 0⁰，torch 返回 1，所以 D(1) = P_K 成立。单元测试会检查这一点。
- **循环残差的推导：** 锚点 0 的像素 p 在 τ = 1 时到达 y = p + P_K(p)，这是 I1 上的一个点。如果两组轨迹描述同一个运动，那么从 y 出发的锚点 1 轨迹在任何时刻都应该和 p 的轨迹重合：`X₁(y, 1 − s) = X₀(p, s)`，于是 `r = [y + D₁(y, 1 − s)] − [p + D₀(p, s)] = P_K(p) + D₁(y, 1 − s) − D₀(p, s)`。s = 0 时它就是普通的前后向一致性误差；s = 1 时恒为 0，所以 `cyc_times` 里不放 1。
- 遮挡像素（p 在 I1 里被挡住）的残差没有意义。估计器里把它当作输入特征，网络自己学会忽略；损失里用前后向检查的掩码排除。
- **以后可以换的：** K 取更大（如 6–8）时，Bernstein 基的每个控制点都影响整段轨迹，可以换成均匀三次 B 样条（局部支撑）。接口不变，只改 `weights`；要保证 D(0) = 0。

### 5.3 `models/traj/encoders.py`：三个编码器

**负责：** 把图像和事件编码成 1/8 分辨率的特征。

| 类 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `MatchFeatEncoder` | if_4 (N,128,h4,w4) | (N,C,h8,w8) | 图像相关体：I_ref 与 I_oth |
| `EventSliceEncoder` | 归一化 voxel (N,NB,H,W) | feats (N,S,C,h8,w8)、times (S,) | 事件相关体：参考窗与目标窗 |
| `ContextEncoder` | im_ref、im_oth (N,3,H,W)，vox (N,NB,H,W) | net、inp (N,128,h8,w8) | GRU 的初始隐状态和每次迭代的上下文 |

**参考：** RAFT `core/extractor.py`（BSD-3）的“特征网络 + 上下文网络”分工；BFlow `raft.py` 的 `gen_voxel_grids`（把 voxel 切成时间窗分别过特征网络）。

`models/traj/encoders.py`：

```python
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
```

**要点：**

- `MatchFeatEncoder` 复用 OF-AF ImageEncoder 的 `if_4`，省计算，而且这个编码器同时被 BiW 的重建损失训练。如果发现图像相关体的质量差（例如端点光度损失降不下去），换成独立的 RAFT `BasicEncoder`（在原图上算、instance norm、输出 256 通道），其余不变。
- `EventSliceEncoder` 默认 S = 9、win = 16、NB = 128：窗起点是 0, 14, 28, …, 112，中心时间是 0.059, 0.169, …, 0.941。起点关于时间反转对称，所以锚点 1（输入 `-voxel.flip(1)`）的第 k 个窗正好是原来的第 S−1−k 个窗。
- 参考窗（第 0 个窗）的中心时间是 0.059，不是 0。所以估计器在事件相关体里查表时用的是**相对位移** `D(s_k) − D(s_0)`，而不是 `D(s_k)`（见 §5.6）。
- 事件相关体的含义：参考窗里的事件来自 t≈0 时的边缘，目标窗里的事件来自 t≈s_k 时的同一条边缘，边缘沿轨迹移动，所以在 `p + D(s_k) − D(s_0)` 处相关最大。这是 BFlow 的核心思路，在真实事件（DSEC）上有效。
- 不用 BatchNorm：batch 通常是 1，而且锚点 0 和 1 拼在同一个 batch 里，BN 会把两者的统计量混在一起。

### 5.4 `models/traj/corr.py`：相关体

**负责：** 建相关体，并在给定坐标附近取 (2r+1)² 的局部相关值。

| 类 | 构造 | 调用 | 输出 |
| --- | --- | --- | --- |
| `CorrBlock(fmap1, fmap2, L, r)` | fmap1、fmap2 (N,C,h,w) | `(coords)`，coords (N,2,h,w) 是 fmap2 上的绝对像素坐标 | (N, L·(2r+1)², h, w) |
| `AltCorrBlock(...)` | 同上 | 同上 | 同上，数值相同 |

**参考：** RAFT `core/corr.py` 的 `CorrBlock`（BSD-3-Clause）。RAFT 的 `AlternateCorrBlock` 依赖 CUDA 扩展 `alt_cuda_corr`，这里的 `AltCorrBlock` 是纯 PyTorch 写的，不需要编译。

`models/traj/corr.py`：

```python
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
```

**要点：**

- 通道排列：先按金字塔层，层内按窗口行优先（dy 外层、dx 内层）。`CorrBlock` 和 `AltCorrBlock` 的排列和数值都相同（池化和点积可以交换顺序），所以可以用 full 训练、推理大图时换成 alt，不用重新训练。单元测试会检查两者相等。
- 显存：全对相关体每个目标是 N·(h8·w8)² 个 float。训练时（256×256 裁剪）可以忽略；GOPRO 全图 1280×720 推理时约 11 GB，所以推理大图用 `--corr_impl alt`。具体数字见 §10。
- 金字塔的最顶层尺寸至少要 2：L = 4 时要求 h8、w8 ≥ 8，也就是输入至少 64×64。

### 5.5 `models/traj/update.py`：迭代更新

**负责：** 一次迭代里，根据相关特征、当前控制点和循环残差，预测控制点增量和凸上采样掩码。

| 类 / 函数 | 输入 | 输出 |
| --- | --- | --- |
| `TrajMotionEncoder` | ctrl_flat (N,2K,h8,w8)、cyc (N,2M+1,h8,w8)、corr (N,corr_ch,h8,w8) | (N,128,h8,w8) |
| `TrajUpdateBlock` | net、inp、corr、ctrl_flat、cyc | net、Δctrl (N,2K,h8,w8)、mask (N,9·4,h8,w8) |
| `convex_upsample(x, mask, r)` | x (N,C,h,w)、mask (N,9r²,h,w) | (N,C,rh,rw)，数值 ×r |

**参考：** RAFT `core/update.py` 的 `SepConvGRU`、`BasicMotionEncoder`、`BasicUpdateBlock`，以及 `core/raft.py` 的 `upsample_flow`（BSD-3）；控制点输出头参考 bflow `models/raft_spline/update.py` 的 `BezierHead`（Apache-2.0）。

`models/traj/update.py`：

```python
"""迭代更新模块。SepConvGRU / 运动编码器 / 凸上采样改写自 RAFT core/update.py、core/raft.py
（BSD-3-Clause），控制点输出头参考 uzh-rpg/bflow models/raft_spline/update.py 的 BezierHead（Apache-2.0）。"""
import torch
import torch.nn.functional as F
from torch import nn


class SepConvGRU(nn.Module):
    """与 RAFT 相同：先 1x5 再 5x1 的可分离卷积 GRU。"""

    def __init__(self, hidden_dim=128, input_dim=256):
        super().__init__()
        cin = hidden_dim + input_dim
        self.convz1 = nn.Conv2d(cin, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(cin, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(cin, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(cin, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(cin, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(cin, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        for cz, cr, cq in ((self.convz1, self.convr1, self.convq1), (self.convz2, self.convr2, self.convq2)):
            hx = torch.cat([h, x], 1)
            z, r = torch.sigmoid(cz(hx)), torch.sigmoid(cr(hx))
            q = torch.tanh(cq(torch.cat([r * h, x], 1)))
            h = (1 - z) * h + z * q
        return h


class TrajMotionEncoder(nn.Module):
    """RAFT BasicMotionEncoder 的扩展：除了相关特征和当前“光流”（这里是 2K 个控制点），
    还输入跨锚点循环残差（cyc）。输出 out_ch 通道（最后 2K 通道是控制点本身）。"""

    def __init__(self, corr_ch, K, cyc_ch, out_ch=128):
        super().__init__()
        self.convc1 = nn.Conv2d(corr_ch, 256, 1)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2 * K + cyc_ch, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(192 + 64, out_ch - 2 * K, 3, padding=1)

    def forward(self, ctrl_flat, cyc, corr):
        c = F.relu(self.convc2(F.relu(self.convc1(corr))))
        f = F.relu(self.convf2(F.relu(self.convf1(torch.cat([ctrl_flat, cyc], 1)))))
        return torch.cat([F.relu(self.conv(torch.cat([c, f], 1))), ctrl_flat], 1)


class TrajUpdateBlock(nn.Module):
    """一次迭代：运动编码 → GRU → 控制点增量 (2K) + 凸上采样掩码 (9*up*up)。"""

    def __init__(self, corr_ch, K, cyc_ch, hidden=128, ctx=128, up=2):
        super().__init__()
        self.encoder = TrajMotionEncoder(corr_ch, K, cyc_ch, 128)
        self.gru = SepConvGRU(hidden, ctx + 128)
        self.head = nn.Sequential(nn.Conv2d(hidden, 256, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 2 * K, 3, padding=1))
        self.mask = nn.Sequential(nn.Conv2d(hidden, 256, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 9 * up * up, 1))

    def forward(self, net, inp, corr, ctrl_flat, cyc):
        motion = self.encoder(ctrl_flat, cyc, corr)
        net = self.gru(net, torch.cat([inp, motion], 1))
        return net, self.head(net), 0.25 * self.mask(net)      # 0.25：RAFT 的梯度平衡系数


def convex_upsample(x, mask, r):
    """RAFT upsample_flow 的通用版本：x (N,C,h,w)、mask (N,9*r*r,h,w) → (N,C,r*h,r*w)，数值同时乘 r。
    控制点是像素单位的位移，所以和光流一样要乘 r；对 x 是线性的，先上采样控制点再求位移 = 先求位移再上采样。"""
    n, c, h, w = x.shape
    mask = mask.view(n, 1, 9, r, r, h, w).softmax(dim=2)
    up = F.unfold(r * x, [3, 3], padding=1).view(n, c, 9, 1, 1, h, w)
    up = (mask * up).sum(dim=2)                                    # (N,C,r,r,h,w)
    return up.permute(0, 1, 4, 2, 5, 3).reshape(n, c, r * h, r * w)
```

**要点：**

- `TrajMotionEncoder` 就是 RAFT 的 `BasicMotionEncoder`，只是“光流”换成了 2K 个控制点，并多拼了循环残差 `cyc`。
- 掩码乘 0.25 是 RAFT 原有的做法，用来平衡梯度。
- 凸上采样对控制点是线性的，所以“先上采样控制点再求位移”等于“先求位移再上采样”，查询任意时刻都只需上采样一次。
- 如果训练初期控制点跳得很厉害，可以把 `head` 最后一层的权重初始化为 0（`nn.init.zeros_`），让初始增量为 0。

### 5.6 `models/traj/estimator.py`：双锚点轨迹估计器（核心）

**负责：** 从两帧和事件估计两组控制点。

**输入：**

| 名称 | 形状 | 说明 |
| --- | --- | --- |
| `im0`, `im1` | (B,3,H,W) | [0,1]，H、W 是 8 的倍数 |
| `voxel` | (B,NB,H,W) | 原始 voxel（函数内部会归一化） |
| `if0_4`, `if1_4` | (B,128,H/4,W/4) | OF-AF ImageEncoder 的输出 |
| `iters` | int 或 None | 迭代次数，None 时用构造参数 |

**输出：**

| 名称 | 形状 | 说明 |
| --- | --- | --- |
| `preds` | list，每项 (2B,K,2,H/4,W/4) | 训练时每次迭代一项，推理时只有最后一项。前 B 个是锚点 0，后 B 个是锚点 1；局部时间；1/4 像素单位 |
| `metric` | (2B,1,H/4,W/4) | 取值 [−3, 3]，供 v2 的 splat 反转决定前景 |

**参考：** bflow `models/raft_spline/raft.py` 的 `RAFTSpline.forward`（Apache-2.0）：多个时刻的轨迹索引查相关 + GRU 迭代更新 Bézier 控制点。和 BFlow 的区别：(1) 两个锚点拼成 2B 的 batch 共享权重；(2) 每次迭代输入跨锚点循环残差；(3) 没有轨迹真值，监督全部来自 §6 的自监督损失和 BiW 的重建损失。

`models/traj/estimator.py`：

```python
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
```

**每次迭代做了什么：**

1. 用当前控制点算出几个查表时刻的位移：图像相关体在 s = 1 查（I_oth 上 `p + D(1)` 处），事件相关体在目标窗中心时间 s_k 查（`p + D(s_k) − D(s_0)` 处）。
2. 拼起所有查表结果得到 `corr`，通道数 = (1 + 目标窗个数) × L × (2r+1)²，默认 5 × 4 × 49 = 980。
3. 算循环残差特征 `cyc`（2M+1 通道，默认 9）。这里对控制点 detach，残差只作为输入信息，不反传。
4. GRU 更新隐状态，输出控制点增量和上采样掩码；控制点加上增量。
5. 训练时每次迭代都上采样一次存进 `preds`（给序列损失用），推理时只在最后一次上采样。

**要点：**

- **时间反转对称：** 交换两帧并把 voxel 换成 `-voxel.flip(1)` 后，输出的两半正好互换。这是共享权重的直接结果，单元测试会检查；如果这个测试不过，说明 2B batch 的拼法或某处索引写错了。
- `detach_iter=True` 和 RAFT 一样，每次迭代前截断控制点的梯度。BFlow 的代码注释说，对 Bézier 控制点截断能让 DSEC 上的验证 EPE 降低最多 5%，同时还省显存，所以默认截断。
- 迭代次数：训练 6 次，推理先试 8–12 次（RAFT 训练 12 次、评测 24–32 次）。推理时加多迭代是否有益，要在验证集上确认。
- `metric` 用 tanh 限幅到 [−3, 3]，exp 之后在 [0.05, 20]，这样 v2 反转的分母有下界，不会出现 OF-AF 空洞掩码那种梯度尖峰。

**超参数（构造参数）：**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `degree` | 4 | K，控制点个数 |
| `n_slices` / `slice_win` | 9 / 16 | 事件时间窗个数 / 每窗 bin 数 |
| `target_slices` | (2, 4, 6, 8) | 参与查表的目标窗（中心约 0.28、0.5、0.72、0.94） |
| `cyc_times` | (0, 0.25, 0.5, 0.75) | 循环残差的局部时间 |
| `corr_levels` / `corr_radius` | 4 / 3 | 金字塔层数 / 查表半径 |
| `iters` | 6 | 迭代次数 |
| `feat_ch` / `hidden` / `ctx` | 128 / 128 / 128 | 特征、GRU 隐状态、上下文通道数 |
| `corr_impl` | `"full"` | `"alt"` 用于推理大图 |

### 5.7 `models/traj/reversal.py`：轨迹反转

**负责：** 由锚点网格上的 F_{a→t}（就是 D_a(·, t)）得到时刻 t 网格上的 F_{t→a}。

| 函数 | 输入 | 输出 | 用于 |
| --- | --- | --- | --- |
| `invert_fixed_point(disp, n_iter)` | disp (N,2,h,w) | back (N,2,h,w) | v1 的反向光流；v2 的兜底；CM 损失 |
| `splat_coverage(disp)` | disp | (N,1,h,w)，不反传 | v2 的 valid 掩码 |
| `reverse_by_splat(disp, metric)` | disp、metric (N,1,h,w) | back、valid | v2 的主要反转 |
| `complement_from_other(back, valid, end, fallback)` | 都是 (2B,…) | 补完空洞的 back | v2 |

**参考：**

- 不动点求逆：M. Chen et al., “A simple fixed-point approach to invert a deformation field”, Medical Physics, 2008。迭代 p ← x − D(p)，局部雅可比范数小于 1 时收敛。numpy 验证：旋转 + 缩放 + 平移的平滑场，迭代 4 次后落点误差中位数 4.5e-4 像素。
- softmax splatting：Niklaus & Liu, CVPR 2020。OF-AF 自带 `models/softsplat.py`；原作者声明该实现仅限学术用途，商用需要联系作者。

`models/traj/reversal.py`：

```python
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
```

**v1 和 v2 的区别：**

| | v1：`fixed_point` | v2：`splat` |
| --- | --- | --- |
| 做法 | 对每个时刻 t 的像素 x 解 p + D(p) = x | 把 −D(p) 用 softmax splat 推到 p + D(p) |
| 遮挡（多个 p 落到同一个 x） | 收敛到其中一个，不保证是前景 | metric 大的胜出（网络学“谁在前面”） |
| 空洞（没有 p 落到 x） | 照样给出一个值（相当于外推） | valid = 0，用另一锚点补，再不行用 v1 兜底 |
| 是否需要 CUDA | 不需要 | 需要（CuPy） |
| 建议 | 第 1 阶段先用它 | 第 2 阶段再换，做消融 |

**空洞补全的推导（`complement_from_other`）：** 锚点 0 在 x 处有空洞，说明 x 在 I0 里被遮挡或刚露出来。先用锚点 1 的反向光流走到 I1：y = x + F_{t→1}(x)；锚点 1 的像素 y 沿自己的轨迹在 τ = 0 时到达 y + F₁→₀(y)。所以 `F_{t→0}(x) ≈ F_{t→1}(x) + F₁→₀(x + F_{t→1}(x))`。numpy 验证：匀速运动下这个公式精确等于 −t·v。注意 x 在 I0 里本来就看不见，所以补出来的光流只是几何上合理，warp 出来的颜色仍然可能不对；BiW 的空洞掩码和融合权重会让这类像素主要用 I1 的结果。

**要点：** v2 的分母 clamp 到 0.02。覆盖率 ≥ 0.5 的像素上，分母至少是 0.5 × e⁻³ ≈ 0.025，clamp 基本不起作用；它只是保证即使有像素漏过掩码，梯度也有上界。不要用 softsplat 的 `"soft"` 模式，它内部除以 `q + 1e-7`，正是 OF-AF 空洞掩码梯度尖峰的来源。

### 5.8 `models/traj/query.py`：按时刻查询

**负责：** 给定一个查询时刻 τ，生成 BiW 的全部 6 个张量输入。

| 接口 | 输入 | 输出 |
| --- | --- | --- |
| `TrajQuery(basis, reversal)` 的 `forward(ctrl, metric, t)` | ctrl (2B,K,2,h4,w4)、metric (2B,1,h4,w4)、t (B,) | dict：`flow_fwd` (2B,2,h4,w4) = [F₀→τ; F₁→τ]，`flow_bwd` = [Fτ→₀; Fτ→₁]，`valid` (2B,1,h4,w4) |
| `event_feature_at(efs, t)` | efs (B,16,128,h4,w4)、t (B,) | (B,128,h4,w4) |
| `TimeContext` 的 `forward(f4, flow_bwd, ev_t, s, valid)` | f4 (2B,128,h4,w4)、flow_bwd、ev_t (2B,128,h4,w4)、s (2B,) | (2B,128,h4,w4) = [ctxt0; ctxt1] |

**参考：** 本设计。`TimeContext` 替代 OF-AF 里 BidirFlow 每步产出的 `ctxt0`/`ctxt1`；这两个特征在 BiW 里进 `RefBlock`（细化空洞掩码）和 refine 分支。

`models/traj/query.py`：

```python
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
```

**要点：**

- `TrajQuery` 一次处理 2B 个样本：前 B 个用 s = τ，后 B 个用 s = 1 − τ。
- `TimeContext` 的输入：用 Fτ→ₐ 把锚点特征 warp 到时刻 τ、τ 附近的事件特征、反向光流本身、valid、局部时间的正余弦编码，共 128 + 128 + 2 + 1 + 8 = 267 通道。
- `event_feature_at` 把 OF-AF EventsEncoder 的第 g 组近似看成时间 (g + 0.5)/16。EventsEncoder 的 `pre_conv` 会混合所有 bin，所以这只是近似，网络会自己学怎么用。如果发现上下文特征不好，可以改成直接从 voxel 里切出 τ 附近的 bin 再编码。
- 每个 τ 的开销：v1 是一次 einsum + 4 次 grid_sample；TimeContext 在 256×256 训练裁剪上约 7.4 GMAC（两个锚点合计），大约是 BiW（约 42 GMAC）的六分之一。觉得慢可以把 TimeContext 的通道数减到 64。

### 5.9 `models/model_traj.py`：组装

**负责：** 把保留的 OF-AF 模块和新模块接起来。对外接口和 `MyModel` 完全相同，所以训练、评测脚本只要换类名。
| 参数 | 说明 |
| --- | --- |
| `im0, im1, event, interp_times` | 同 MyModel |
| `return_aux` | 同 MyModel：返回 `(preds, fwd0, fwd1, valid0, valid1)`，给 `--fwd_loss_weight` 用 |
| `return_traj` | 新增：额外返回 dict：`ctrl_preds`（每次迭代的控制点）、`metric`、`flow_bwd`（每个 τ 的反向光流列表） |
| `iters` | 覆盖估计器的迭代次数 |

需要的 `args` 字段：`voxel_bins`、`nb_of_flow`、`traj_degree`、`traj_iters`、`traj_reversal`、`fix_warp`（必须为 True）、`fix_mask_grad`、`corr_impl`。

`models/model_traj.py`：

```python
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
```

**要点：**

- 构造时强制要求 `--fix_warp`：轨迹模块用 align_corners=True 约定，BiW 里的 `BackwardWarp` 只有开 `fix_warp` 才和它一致。
- BiW 的 `fix_mask_grad` 参数来自 `ofaf_b1d3741_followup.patch`。没打这个补丁的话，删掉这个参数即可。
- BiW 外面照旧用 `checkpoint(..., use_reentrant=False)` 省显存；轨迹查询和 TimeContext 在 checkpoint 外面，`flow_bwd` 要留着给光度损失用。

---
## 6. 损失函数：`losses/traj_losses.py`

### 6.1 各项一览

| 项 | 公式（简写） | 监督什么 | 用哪次迭代 | 起始权重 | 参考 |
| --- | --- | --- | --- | --- | --- |
| L_rec | 0.1·LPIPS + Charbonnier（原有） | 最终插帧 | 最后一次 | 1 | OF-AF |
| L_fwd | 前向 splat 帧的掩码 Charbonnier（原有 `--fwd_loss_weight`） | F₀→τ、F₁→τ | 最后一次 | 0.1 | 前面的修复 |
| **L_ph** | mean_x min_a ρ(I_a(x + Fτ→ₐ(x)) − I_τ(x))，对全部 GT 时刻取平均 | 反向光流，进而是整条轨迹 | 最后一次 | 0.5 | Monodepth2 的逐像素最小重投影（只借思路） |
| L_end | ρ(I_b(p + P_K(p)) − I_a(p))，前后向一致性掩码 | 两帧端点光流 | 每次（γ 加权） | 0.2 | UnFlow / ARFlow |
| L_cyc | (‖r(p,s)‖² + ε²)^0.45，s ∈ {0, ¼, ½, ¾}，掩码同上 | 两个锚点的轨迹一致 | 每次（γ 加权） | 0.02 | UnFlow 的前后向损失，扩展到连续时间 |
| L_sm | 边缘感知二阶平滑，作用在全部控制点上 | 空间平滑 | 最后一次 | 0.05 | ARFlow `smooth_grad_2nd`（MIT） |
| L_cm | 平均时间戳的平方和 / 有事件的像素数 | 事件沿轨迹对齐 | 最后一次 | 0（第 3 阶段 0.01–0.05） | Zhu et al. 2019；Hagenaars et al. 2021；taming_event_flow（MIT） |

ρ 是 Charbonnier：√(x² + 10⁻⁶)。γ = 0.8，按 RAFT 的 `sequence_loss` 对第 i 次迭代加权 γ^(N−1−i)，再除以权重和，这样换迭代次数时损失的量级不变。

**L_ph 是最重要的一项。** GOPRO 每个训练样本有 31 张 GT 中间帧，等于在 31 个时刻都约束了轨迹，比两帧光流的自监督强得多。时刻 τ 的每个像素至少在 I0、I1 之一里可见，所以逐像素取两边误差的较小值就能自动处理遮挡，不需要显式的遮挡掩码。采样点出界的一边会被加上 1 的惩罚，保证不会被选中。

**L_cm 的含义：** 把第 b 个 bin（时刻 τ_b）的事件沿 Fτ_b→₀ 推回 τ = 0（锚点 1 推到 τ = 1）。轨迹正确时，同一条边缘在不同时刻产生的事件会落到同一个像素上，这个像素里的事件“到锚点的时间距离”有早有晚、平均值较小；轨迹错误时，晚到的事件散落在边缘外侧，那些像素只有晚的时间戳，平均值大。最小化平均时间戳的平方和就是让事件聚焦。Zhu et al. 也是同时以区间起点和终点为参考时刻计算这个损失，两个锚点正好对应这两个参考时刻。这里用 voxel 切片近似逐事件计算：正负极性用 `relu(v)` 和 `relu(−v)` 近似（同一像素同一 bin 里的正负事件已经抵消，所以只是近似）。

### 6.2 代码

`losses/traj_losses.py`：

```python
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
```

### 6.3 权重怎么调

1. 先把各项的原始值（不乘权重）打印出来，看训练开始后约 100 步的均值。`TrajLoss` 返回的字典就是原始值。
2. 按这个比例设权重：加权后的 L_ph 约为 L_rec 的 0.3–0.5 倍，L_end 约 0.1–0.2 倍，L_cyc 和 L_sm 各约 0.05 倍。
3. 看裁剪前的梯度范数（`clip_grad_norm_` 的返回值）。某一项加进来后范数暴涨，就把它的权重减半。
4. 轨迹塌缩成 0 的征兆：L_sm 非常小，L_end 不下降，可视化里控制点几乎全是 0。这时减小 `w_sm`，或者用 `--traj_warmup_iters` 先只训练光流（见 §7.1）。
5. CM 先关掉。第 3 阶段从 0.01 开始加，同时观察“事件塌缩”：IWE 里出现少数特别亮的点、控制点出现大范围的收缩。出现就降权重或去掉。

---

## 7. 训练和评测脚本的改动

### 7.1 `train_model.py`（4 空格缩进）

**(1) 文件开头 import，第 9 行 `from models.model import MyModel` 下面加：**

```python
from models.model_traj import DACTModel
from losses.traj_losses import TrajLoss
```

**(2) 命令行参数，加在 `--fix_voxel` 那一行下面：**

```python
    parser.add_argument('--flow_mode', choices=['bifeb', 'traj'], default='bifeb',
                        help='bifeb = 原 OF-AF；traj = DACT 轨迹光流')
    parser.add_argument('--traj_degree', type=int, default=4, help='控制点个数 K')
    parser.add_argument('--traj_iters', type=int, default=6)
    parser.add_argument('--traj_reversal', choices=['fixed_point', 'splat'], default='fixed_point')
    parser.add_argument('--corr_impl', choices=['full', 'alt'], default='full')
    parser.add_argument('--w_ph', type=float, default=0.5)
    parser.add_argument('--w_end', type=float, default=0.2)
    parser.add_argument('--w_cyc', type=float, default=0.02)
    parser.add_argument('--w_sm', type=float, default=0.05)
    parser.add_argument('--w_cm', type=float, default=0.0)
    parser.add_argument('--traj_warmup_iters', type=int, default=0,
                        help='前 N 步只用轨迹损失（和 fwd_loss），不用重建损失')
```

**(3) 构造模型，第 161 行 `model = MyModel(args)` 改为：**

```python
    model = DACTModel(args) if args.flow_mode == 'traj' else MyModel(args)
```

**(4) `train()` 里，`lps = LPIPS(net="vgg").cuda()` 下面加：**

```python
    traj_loss = (TrajLoss(model.traj.basis, args.w_ph, args.w_end, args.w_cyc, args.w_sm, args.w_cm)
                 if args.flow_mode == 'traj' else None)
    global_step = 0
```

**(5) 前向和解包。原来的这段：**

```python
            use_forward_loss = args.fwd_loss_weight > 0
            output = model(img0, img1, voxel_01, times, return_aux=use_forward_loss)
            if use_forward_loss:
                imst, fwd0, fwd1, valid0, valid1 = output
            else:
                imst = output
            gts = torch.stack(gts.split(3, dim=1), dim=1)
            b, t, c, h, w = imst.shape
```

替换为：

```python
            use_forward_loss = args.fwd_loss_weight > 0
            kw = {'return_traj': True} if traj_loss is not None else {}
            output = model(img0, img1, voxel_01, times, return_aux=use_forward_loss, **kw)
            traj_out = None
            if traj_loss is not None:
                output, traj_out = output
            if use_forward_loss:
                imst, fwd0, fwd1, valid0, valid1 = output
            else:
                imst = output
            gts = torch.stack(gts.split(3, dim=1), dim=1)
            b, t, c, h, w = imst.shape

            # 轨迹损失要用 (B,T,3,H,W) 的 gts，必须放在后面 gts.view(b*t, ...) 之前
            traj_total, traj_logs = imst.new_zeros(()), {}
            if traj_loss is not None:
                traj_total, traj_logs = traj_loss(traj_out['ctrl_preds'], img0, img1, gts,
                                                  traj_out['flow_bwd'], voxel_01)
```

**(6) 总损失。原来的：**

```python
            loss = loss_p.mean() + torch.mean(loss_1) + args.fwd_loss_weight * fwd_loss
```

替换为：

```python
            w_rec = 0.0 if global_step < args.traj_warmup_iters else 1.0
            loss = (w_rec * (loss_p.mean() + torch.mean(loss_1))
                    + args.fwd_loss_weight * fwd_loss + traj_total)
```

**(7) 梯度裁剪和日志。原来的：**

```python
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
```

替换为：

```python
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            global_step += 1
```

`bar.set_postfix({...})` 的字典里加两项：

```python
                             'grad': f'{float(grad_norm):.3f}',
                             **{k: f'{v:.4f}' for k, v in traj_logs.items()},
```

**(8) 保存 checkpoint。`cpk = {...}` 里加一项：**

```python
            'model_config': {'flow_mode': args.flow_mode, 'traj_degree': args.traj_degree,
                             'traj_iters': args.traj_iters, 'traj_reversal': args.traj_reversal,
                             'voxel_bins': args.voxel_bins, 'nb_of_flow': args.nb_of_flow},
```

`training_options` 里把 `w_ph`、`w_end`、`w_cyc`、`w_sm`、`w_cm`、`traj_warmup_iters` 也记上，方便以后查。

### 7.2 训练脚本示例 `train_traj.sh`

```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python train_model.py \
    --epoch 20 \
    --lr 1e-4 \
    --eta_min 1e-6 \
    --batch_size 1 \
    --num_worker 8 \
    --dataset gopro \
    --voxel_bins 128 \
    --nb_of_flow 16 \
    --save_dir train/dact_v1 \
    --flow_mode traj \
    --traj_degree 4 \
    --traj_iters 6 \
    --traj_reversal fixed_point \
    --fix_warp \
    --fix_voxel \
    --fix_mask_grad \
    --lpips_normalize \
    --fwd_loss_weight 0.1 \
    --w_ph 0.5 --w_end 0.2 --w_cyc 0.02 --w_sm 0.05 --w_cm 0 \
    --traj_warmup_iters 2000 \
    --seed 1234 \
    "$@"
```

- `--fix_query` 在 traj 模式下不起作用（没有 Query 模块），可以不加；但评测时的 `--fix_*` 开关要和训练时一致，否则 checkpoint 的标志检查会报错。
- `--fix_mask_grad` 需要先打 `ofaf_b1d3741_followup.patch`。
- 想先在单个场景上冒烟测试：`bash train_traj.sh --gopro_root /tmp/gopro_one --epoch 1 --save_dir train/dact_smoke`。

### 7.3 可选：用官方权重初始化共享模块

`load_weights` 里 `--only_weights` 分支用的是 `strict=False`，所以可以直接：

```bash
bash train_traj.sh --weights_path train/bsergb.pth --only_weights
```

EventsEncoder、ImageEncoder、BiW 会加载官方权重，`traj.*`、`time_ctx.*` 会列在 Missing 里、保持随机初始化。官方权重是在有 bug 的光流下训出来的，这样初始化不一定更好，要和从头训练对比。官方 checkpoint 没有记录 fix 标志，加载时只会打印一条 NOTE。

### 7.4 `bsergb_eval.py`

**(1) import：**

```python
from models.model_traj import DACTModel
```

**(2) 参数（加在 `--allow_config_mismatch` 下面）：**

```python
    parser.add_argument('--flow_mode', choices=['bifeb', 'traj'], default='bifeb')
    parser.add_argument('--traj_degree', type=int, default=4)
    parser.add_argument('--traj_iters', type=int, default=6)
    parser.add_argument('--traj_reversal', choices=['fixed_point', 'splat'], default='fixed_point')
    parser.add_argument('--corr_impl', choices=['full', 'alt'], default='full',
                        help='大图推理用 alt，显存小很多，结果相同')
    parser.add_argument('--traj_iters_eval', type=int, default=None, help='推理时的迭代次数，默认同训练')
```

**(3) 构造模型。原来是先 `model = MyModel(args)` 再 `state = torch.load(...)`，改成先读 checkpoint、用里面记录的结构参数建模型：**

```python
    state = torch.load(args.checkpoint, map_location='cpu')
    for k, v in state.get('model_config', {}).items():
        setattr(args, k, v)                 # 结构参数以 checkpoint 为准
    model = DACTModel(args) if args.flow_mode == 'traj' else MyModel(args)
```

原来的标志检查和 `load_state_dict` 保持不变（删掉后面重复的那行 `state = torch.load(...)`）。`model.cuda()` 之后加：

```python
    if args.flow_mode == 'traj' and args.traj_iters_eval:
        model.traj.iters = args.traj_iters_eval
```

**(4) 补边到 8 的倍数。`pad()` 里：**

```python
        h_pad = (4 - h % 4) % 4
        w_pad = (4 - w % 4) % 4
```

改为：

```python
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
```

预测结果之后会裁回原尺寸，所以这个改动对原模型的评测没有影响（只是多补几行零）。BS-ERGB 970×625 会补到 976×632。

### 7.5 GOPRO 评测和“非网格时刻”

仓库里没有 GOPRO 的评测入口，但 `datasets/gopro.py` 里有 `GOPRO_Test`（默认 skips = [7, 15]，720×1280 本身就是 8 的倍数，不需要补边）。照 `bsergb_eval.py` 的写法建模型、加载 checkpoint，然后调用 `GOPRO_Test(args.voxel_bins, skips=[7, 15], test_root=..., fix_voxel=args.fix_voxel).test(model)` 即可。注意两点：

- `GOPRO_Test` 里还没有 `bsergb_eval.py` 已有的几项评测修复：LPIPS 输入没有从 [0,1] 换到 [−1,1]，FloLPIPS 权重是必需的。要么照 `bsergb_eval.py` 补上，要么先只看 PSNR/SSIM。
- `GOPRO_Test` 只评 i = 1, 3, 5, …，对应时刻 (i+1)/(skip+1)。skip = 7、15 时这些时刻全落在 1/16 网格上，OF-AF 的离散节点在这里正好不用插值。**要体现连续轨迹的优势，加 skip = 6**（时刻 k/7，不在网格上），或者评测全部 i。

---

## 8. 分阶段实施计划

和基线比较时要公平：OF-AF 基线也要用同样的修复（`--fix_query --fix_warp --fix_voxel --fix_mask_grad --fwd_loss_weight 0.1 --lpips_normalize`）、同样的数据和训练轮数。

| 阶段 | 内容 | 预计 | 通过标准 |
| --- | --- | --- | --- |
| 0（可选） | 在现有 OF-AF 里加跨分支一致性损失：F₀→₁(a) ≈ b − a，a = x + Fτ→₀(x)，b = x + Fτ→₁(x)（用 `flows0t[16]`） | 约 1 周 | 用最小代价检验“轨迹一致性有帮助”这个假设；有提升再继续 |
| 1a | 写完 §5 全部文件，跑通 §9.1 的单元测试 | 3–5 天 | 10 个测试全部通过 |
| 1b | 合成数据（§9.3）上只训练轨迹（`--traj_warmup_iters` 设成总步数，即只用轨迹损失） | 3–5 天 | τ = 0.5 时的轨迹 EPE 明显低于“两帧光流 × τ”的线性基线；前景（弯曲轨迹）上差距更大 |
| 1c | 完整模型在单个样本上过拟合 | 1–2 天 | 几百步内重建 PSNR > 40 dB；轨迹可视化平滑 |
| 1d | GOPRO 完整训练（v1：不动点反转） | 1–2 周 | 训练稳定（梯度范数没有尖峰）；skip 7/15 的 PSNR 不低于公平基线；skip 6 上优于基线 |
| 2 | 换成 v2 反转（`--traj_reversal splat`），做 v1/v2 消融 | 1–2 周 | 遮挡边界的误差下降（看误差图），整体 PSNR 不降 |
| 3 | 加 CM 损失（`--w_cm 0.01` 起），然后在 BS-ERGB 上微调 | 2–3 周 | 事件 IWE 更锐利；BS-ERGB 指标提升；没有事件塌缩 |
| 4（可选） | K ∈ {2, 3, 4, 6}、B 样条基、截断与否、相关目标窗个数等消融 | 按需 | – |

BS-ERGB 是真实事件，噪声和热像素比 V2E 多。第 3 阶段之前先用 `check_events.py` 看一下事件分布；必要时在 CM 损失里忽略计数很小的像素（调大 `den_thr`）。

---

## 9. 测试与验证

### 9.1 单元测试：`tests/test_traj.py`

```bash
python -m pytest -q tests/test_traj.py
```

前 8 个在 CPU 上就能跑（不需要 CuPy），后 2 个需要 GPU。

| 测试 | 覆盖文件 | 检查什么 |
| --- | --- | --- |
| `test_basis_endpoints_and_linear_precision` | basis.py | D(0) = 0、D(1) = P_K、匀速精确表示 |
| `test_cycle_residual_zero_for_consistent_anchors` | basis.py、common.py | 一致的两组轨迹残差为 0 |
| `test_fixed_point_inverse_affine` | reversal.py | 仿射场求逆后落点误差 < 0.01 像素 |
| `test_convex_upsample_constant_interior` | update.py | 常数场上采样后内部 = 2 倍原值 |
| `test_corr_full_equals_alt` | corr.py | 两种相关实现数值相同、通道排列相同 |
| `test_corr_zero_motion_peak` | corr.py | 零位移时窗口中心最相关 |
| `test_estimator_shapes_and_time_reversal_symmetry` | encoders.py、estimator.py | 输出形状；交换两帧 + 时间反转后两半互换 |
| `test_estimator_gradients_reach_all_parts` | estimator.py 全部子模块 | 每个子模块都有梯度（防止 OF-AF 那种“某分支没梯度”的问题） |
| `test_splat_reversal_translation`（GPU） | reversal.py | 平移场的 splat 反转 = −v |
| `test_full_model_forward_backward`（GPU） | model_traj.py、traj_losses.py | 整个模型前向、反向，损失和梯度都是有限值 |

`tests/test_traj.py`：

```python
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
```

### 9.2 单样本过拟合

从 GOPRO 训练集取一个样本，固定裁剪、关掉翻转，用 `--lr 1e-4` 训练 500 步（所有损失都开）。期望：重建损失持续下降，PSNR 超过 40 dB；L_ph、L_end 同步下降；控制点可视化（§9.4）平滑、没有噪点。降不下去优先查：坐标约定（是否开了 `--fix_warp`）、`gts` 有没有在 view 之前传给 TrajLoss、voxel 有没有对齐到帧区间（`--fix_voxel`）。

### 9.3 合成数据：有轨迹真值的验证

真实数据集没有轨迹真值，只能看插帧指标。`tools/make_synthetic_traj.py` 生成和 GOPRO 训练集布局相同的场景：背景做“旋转 + 缩放 + 加速平移”，前景方块沿圆周运动并遮挡背景，事件用简化的 DVS 模型（对数亮度变化每超过阈值 C 发一个事件）仿真。轨迹真值用 `gt_displacement()` 解析计算。

```bash
for s in 0 1 2 3 4 5 6 7; do
  python tools/make_synthetic_traj.py --out /data/synth_traj/train/scene$s --frames 64 --speed 0.3 --seed $s
done
python tools/make_synthetic_traj.py --out /data/synth_traj/test/scene0 --frames 64 --speed 0.3 --seed 100
```

实测（--speed 0.3，skip 31 的 32 帧窗口）：位移中位数 31 像素、最大 69 像素；用真值 warp 的误差 0.57/255；区间中点处轨迹偏离直线的中位数：背景 2.9 像素、前景 5.0 像素。所以“线性插值”和“真实轨迹”在这里有明显差距，可以检验连续轨迹是否学到了弯曲运动。

`tools/make_synthetic_traj.py`：

```python
"""生成带轨迹真值的合成场景（目录布局与 GOPRO 训练集相同：images/*.png + events/*.npz，每个帧间隔一个事件文件）。
背景做“旋转+缩放+加速平移”，前景方块沿圆周运动并遮挡背景，所以帧间轨迹是弯的、有遮挡。
gt.npz 存各帧前景掩码和尺寸参数；任意时刻的轨迹真值用 gt_displacement() 解析计算。

    python tools/make_synthetic_traj.py --out /data/synth_traj/scene0 --frames 40
"""
import argparse
import os

import cv2
import numpy as np

FG = 64   # 前景方块边长


def texture(h, w, rng):
    t = np.zeros((h, w, 3), np.float32)
    for s in (4, 8, 16, 32, 64):                     # 多尺度噪声，保证到处有纹理
        n = rng.random((h // s + 2, w // s + 2, 3)).astype(np.float32)
        t += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) / s ** 0.5
    t -= t.min()
    return t / t.max()


def similarity(angle, scale, tx, ty, cx, cy):
    """纹理坐标 → 图像坐标的 3x3 矩阵：绕 (cx,cy) 旋转缩放，再平移 (tx,ty)。"""
    c, s = np.cos(angle) * scale, np.sin(angle) * scale
    return np.array([[c, -s, cx - c * cx + s * cy + tx], [s, c, cy - s * cx - c * cy + ty], [0, 0, 1.0]])


def mats(g, h, w, speed=1.0):
    """全局时间 g（单位：帧）时的背景矩阵 B 与前景矩阵 F。speed 整体缩放运动快慢。"""
    g = g * speed
    Bm = similarity(0.01 * g + 0.0008 * g * g, 1 + 0.004 * g, -w / 2 - 1.5 * g - 0.08 * g * g, -h / 2 + 0.8 * g, w, h)
    ang = 0.08 * g
    Fm = similarity(ang, 1.0, w / 2 - FG / 2 + 70 * np.cos(ang), h / 2 - FG / 2 + 50 * np.sin(ang), FG / 2, FG / 2)
    return Bm, Fm


def render(bg_tex, fg_tex, g, h, w, speed):
    Bm, Fm = mats(g, h, w, speed)
    img = cv2.warpAffine(bg_tex, Bm[:2], (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fg = cv2.warpAffine(fg_tex, Fm[:2], (w, h), flags=cv2.INTER_LINEAR)
    m = cv2.warpAffine(np.ones((FG, FG), np.float32), Fm[:2], (w, h), flags=cv2.INTER_LINEAR)
    return img * (1 - m[..., None]) + fg * m[..., None], m > 0.5


def gt_displacement(fg_mask_k, k, g, h, w, speed=1.0):
    """帧 k 的像素 p 在全局时间 g 的位移真值 (2,H,W)：前景像素 x = F(g)F(k)^-1 p，其余 x = B(g)B(k)^-1 p。
    评测时：锚点 I0 = 帧 k、I1 = 帧 k+skip+1，物理时间 tau 对应 g = k + tau*(skip+1)。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    p = np.stack([xx, yy, np.ones_like(xx)]).reshape(3, -1)
    (Bk, Fk), (Bg, Fg) = mats(k, h, w, speed), mats(g, h, w, speed)
    xb = (Bg @ np.linalg.inv(Bk) @ p)[:2].reshape(2, h, w)
    xf = (Fg @ np.linalg.inv(Fk) @ p)[:2].reshape(2, h, w)
    return np.where(fg_mask_k[None], xf, xb) - np.stack([xx, yy])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--size", type=int, nargs=2, default=(256, 320), metavar=("H", "W"))
    ap.add_argument("--sub", type=int, default=32, help="每个帧间隔的事件仿真子步数")
    ap.add_argument("--C", type=float, default=0.15, help="对数亮度阈值")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="运动快慢；GOPRO_Train 固定 skip=31（窗口 32 帧），用 0.3 左右让窗口内位移不超过 ~60 像素")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    h, w = a.size
    bg_tex, fg_tex = texture(2 * h, 2 * w, rng), texture(FG, FG, rng)
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "events"), exist_ok=True)

    masks, log_ref = [], None
    for k in range(a.frames):
        img, m = render(bg_tex, fg_tex, float(k), h, w, a.speed)
        cv2.imwrite(os.path.join(a.out, "images", f"{k:06d}.png"),
                    cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        masks.append(m)
        if k == a.frames - 1:
            break
        xs, ys, ts, ps = [], [], [], []
        for j in range(a.sub + 1):                   # 简化的 DVS 模型：对数亮度变化每超过 C 发一个事件
            gray = cv2.cvtColor(render(bg_tex, fg_tex, k + j / a.sub, h, w, a.speed)[0], cv2.COLOR_RGB2GRAY)
            L = np.log(gray + 1e-2)
            if log_ref is None:
                log_ref = L.copy()
            if j == 0:
                continue
            d = L - log_ref
            n = np.floor(np.abs(d) / a.C).astype(int)
            yy, xx = np.nonzero(n)
            cnt, sgn = n[yy, xx], np.sign(d[yy, xx])
            log_ref[yy, xx] += sgn * cnt * a.C
            rep = np.repeat(np.arange(len(yy)), cnt)
            frac = (np.arange(len(rep)) - np.repeat(np.cumsum(cnt) - cnt, cnt) + 1) / (cnt[rep] + 1)
            xs.append(xx[rep]); ys.append(yy[rep]); ps.append(sgn[rep])
            ts.append(k + (j - 1 + frac) / a.sub)    # 时间单位：帧；子步内均匀分布
        x, y, t, p = (np.concatenate(v) for v in (xs, ys, ts, ps))
        o = np.argsort(t, kind="stable")
        np.savez(os.path.join(a.out, "events", f"{k:06d}.npz"),
                 x=x[o].astype(np.float64), y=y[o].astype(np.float64), t=t[o], p=p[o].astype(np.float64))
    np.savez(os.path.join(a.out, "gt.npz"), fg_mask=np.stack(masks), size=np.array([h, w]), speed=a.speed)
    print("done:", a.out)


if __name__ == "__main__":
    main()
```

训练用 `--dataset gopro --gopro_root /data/synth_traj/train`（`GOPRO_Train` 固定 skip = 31，所以每个场景至少要 33 帧，上面用了 64 帧）。评测轨迹误差的片段：

```python
import numpy as np
import torch
from models.traj.common import upsample_flow
from tools.make_synthetic_traj import gt_displacement

# im0 = 第 k 帧，im1 = 第 k+skip+1 帧，voxel 用和训练相同的方式从这两帧之间的事件文件生成
gt = np.load(scene + "/gt.npz")
h, w = gt["size"]
model.eval()
with torch.no_grad():
    _, traj = model(im0, im1, voxel, times, return_traj=True)
ctrl = traj["ctrl_preds"][-1]
for tau in (0.25, 0.5, 0.75):
    s = torch.tensor([tau], device=ctrl.device)
    pred = upsample_flow(model.traj.basis.displacement(ctrl[:1], s), 4)[0].cpu().numpy()
    lin = tau * upsample_flow(ctrl[:1, -1], 4)[0].cpu().numpy()          # 线性基线：两帧光流 × τ
    g = gt_displacement(gt["fg_mask"][k], k, k + tau * (skip + 1), h, w, float(gt["speed"]))
    epe = lambda d: np.sqrt(((d - g) ** 2).sum(0))
    fg = gt["fg_mask"][k]
    print(f"tau={tau}: EPE 全部 {epe(pred).mean():.2f} / 前景 {epe(pred)[fg].mean():.2f}；"
          f"线性基线 {epe(lin).mean():.2f} / {epe(lin)[fg].mean():.2f}")
```

### 9.4 可视化

```python
import cv2
import numpy as np
import torch
from models.traj.common import upsample_flow


def draw_trajectories(img0_uint8, ctrl, basis, step=16, n_t=9):
    """在 I0 上每隔 step 个像素画一条锚点 0 的轨迹（红→绿表示 τ 从 0 到 1）。ctrl: (2B,K,2,h4,w4)。"""
    taus = torch.linspace(0, 1, n_t, device=ctrl.device)
    d = torch.stack([upsample_flow(basis.displacement(ctrl[:1], t.view(1)), 4)[0] for t in taus]).cpu().numpy()
    out = img0_uint8.copy()
    H, W = out.shape[:2]
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            pts = np.stack([x + d[:, 0, y, x], y + d[:, 1, y, x]], 1)
            for i in range(n_t - 1):
                c = (0, int(255 * i / (n_t - 2)), int(255 * (1 - i / (n_t - 2))))
                cv2.line(out, tuple(np.round(pts[i]).astype(int)), tuple(np.round(pts[i + 1]).astype(int)), c, 1)
    return out
```

另外建议看：τ = 0.25 / 0.5 / 0.75 的 F₀→τ 颜色图（不同 τ 之间应该连续变化）；循环残差 ‖r‖ 的热力图（应该只在遮挡区域大）；第 3 阶段看 CM 前后的 IWE。

### 9.5 正式实验与消融

| 实验 | 目的 |
| --- | --- |
| GOPRO skip 7 / 15（官方协议） | 和论文、公平基线比较 |
| GOPRO skip 6（非网格时刻） | 检验连续轨迹的优势 |
| BS-ERGB skip 1 / 3 | 真实事件 |
| 光流代理指标：非遮挡像素上 warp(I0, Fτ→₀) 与 I_τ 的 PSNR | 不依赖 BiW，单独看光流质量 |
| 合成数据轨迹 EPE（全部 / 前景） | 唯一有真值的光流指标 |
| 消融：K、v1/v2 反转、去掉 L_cyc、去掉循环残差输入、去掉 L_cm、共享/不共享锚点权重、迭代次数 | 每个设计各自的贡献 |
| 速度和显存：`torch.cuda.max_memory_allocated()`、每步耗时 | 和 OF-AF 对比 |

---

## 10. 计算量与显存（粗算）

**全对相关体的显存**（每个目标是 2B·(h8·w8)² 个 float32；默认 5 个目标；金字塔约再加 1/3）：

| 场景 | h8 × w8 | 单个目标 | 5 个目标 + 金字塔 | 建议 |
| --- | --- | --- | --- | --- |
| 训练裁剪 256×256，B = 1 | 32 × 32 | 8.4 MB | 约 56 MB | full |
| BS-ERGB 976×632（补边后），B = 1 | 79 × 122 | 743 MB | 约 4.9 GB | full 可用，显存紧就 alt |
| GOPRO 1280×720，B = 1 | 90 × 160 | 1.66 GB | 约 11 GB | alt |

`AltCorrBlock` 不存体积，每次查表的临时张量是 2B·C·hw·(2r+1)²：GOPRO 全图第 0 层约 2 × 128 × 14400 × 49 ≈ 1.8 亿个 float（约 720 MB，推理时不存梯度）。还嫌大就把 hw 分块。

**计算量**（只数卷积的乘加，256×256 训练裁剪，T = 31，B = 1，手工估算）：

| 部分 | OF-AF | DACT |
| --- | --- | --- |
| 光流估计 | BidirFlow 每步约 19 GMAC × 32 步（16 步 × 2 个方向）≈ 600 GMAC | 编码器约 30 GMAC + 6 次迭代 × 约 6.6 GMAC ≈ 70 GMAC |
| 每个查询时刻 | Query：很小 | TimeContext 约 7.4 GMAC × 31 ≈ 230 GMAC |
| BiW | 约 42 GMAC × 31 ≈ 1300 GMAC | 相同 |
| 合计 | 约 1900 GMAC | 约 1600 GMAC |

这些数字没算 grid_sample、softsplat 和相关体本身，只用来说明量级相当。实际速度和显存以 §9.5 的实测为准。

---

## 11. 风险与对策

| 风险 | 表现 | 对策 |
| --- | --- | --- |
| K = 4 的 Bernstein 曲线表示不了剧烈变化的运动（反弹、急停） | 合成数据前景 EPE 降不下去；运动剧烈的场景插帧变差 | 增大 K；或换成均匀三次 B 样条（局部支撑，§5.2） |
| 事件太稀疏，事件相关体没信息 | 去掉事件相关体（只留图像端点）后结果几乎不变 | 先用 `check_events.py` 看每个 bin 的事件数；V2E 数据用更高的插帧倍率重新仿真 |
| 不动点求逆在大形变、遮挡处不收敛 | 遮挡边界附近的 Fτ→ₐ 抖动 | 主要靠 BiW 的空洞掩码和融合；第 2 阶段换 v2 |
| CM 损失导致事件塌缩 | IWE 出现少数极亮点；控制点大范围收缩 | 权重从 0.01 起；用平均时间戳形式（不用方差）；以光度损失为主 |
| 真实数据（BS-ERGB）上亮度不恒定 | L_ph、L_end 降不下去 | 在 Charbonnier 外再加 census（ARFlow `TernaryLoss`，MIT） |
| 推理大图显存不够 | OOM | `--corr_impl alt`；fp16 存相关体 |
| splat 归一化的梯度尖峰（OF-AF 的老问题） | 梯度范数偶发暴涨 | 覆盖率掩码 detach、分母 clamp、metric 限幅，已在代码里处理；训练时盯着 `grad` 日志 |
| 从头训练收敛慢 | 前几个 epoch PSNR 远低于基线 | `--traj_warmup_iters` 先训光流；或 §7.3 用官方权重初始化共享模块 |

---

## 12. 参考

### 12.1 代码仓库

| 仓库 | 许可证 | 用到的文件 / 函数 | 在本方案中的用途 |
| --- | --- | --- | --- |
| [uzh-rpg/bflow](https://github.com/uzh-rpg/bflow) | Apache-2.0 | `models/raft_spline/bezier.py`（`BezierCurves`）、`raft.py`（`RAFTSpline.forward`、`gen_voxel_grids`）、`update.py`（`BezierHead`、`BasicUpdateBlock`）、`models/raft_utils/corr.py` | 轨迹表示、多时刻查表、迭代更新的总体结构 |
| [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | BSD-3-Clause | `core/corr.py`（`CorrBlock`）、`core/update.py`（`SepConvGRU`、`BasicMotionEncoder`）、`core/raft.py`（`upsample_flow`）、`core/utils/utils.py`（`bilinear_sampler`、`coords_grid`）、`train.py`（`sequence_loss`） | 相关体、GRU、凸上采样、序列损失 |
| [tudelft/taming_event_flow](https://github.com/tudelft/taming_event_flow) | MIT | `utils/iwe.py`（`event_propagation`、`get_interpolation`、`compute_pol_iwe`）、`loss/flow.py`（`iwe_formatting`、`focus_loss`） | CM 损失的公式和归一化方式 |
| [lliuz/ARFlow](https://github.com/lliuz/ARFlow) | MIT | `losses/loss_blocks.py`（`smooth_grad_2nd`、`TernaryLoss`、`SSIM`）、`utils/warp_utils.py`（`get_occu_mask_bidirection`） | 平滑损失、前后向遮挡掩码 |
| [sniklaus/softmax-splatting](https://github.com/sniklaus/softmax-splatting) | 仅限学术用途（README 声明） | `softsplat.py`（OF-AF 已自带 `models/softsplat.py`） | v2 反转、CM 损失 |
| [tub-rip/event_based_optical_flow](https://github.com/tub-rip/event_based_optical_flow) | GPL-3.0 | `src/costs/` | 只读来理解 CM 的各种代价函数，**不要复制代码**（GPL 会传染） |
| [nianticlabs/monodepth2](https://github.com/nianticlabs/monodepth2) | 非商用许可 | – | 只借“逐像素最小重投影”的思路，不复制代码 |

本文代码里改写自 RAFT（BSD-3）和 BFlow（Apache-2.0）的部分，发布时请在对应文件开头保留原仓库的版权和许可证声明。

### 12.2 论文

- M. Gehrig, M. Muglikar, D. Scaramuzza. *Dense Continuous-Time Optical Flow from Event Cameras.* TPAMI 2024.（BFlow）
- Z. Teed, J. Deng. *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow.* ECCV 2020.
- S. Niklaus, F. Liu. *Softmax Splatting for Video Frame Interpolation.* CVPR 2020.
- M. Chen, W. Lu, Q. Chen, K. Ruchala, G. Olivera. *A simple fixed-point approach to invert a deformation field.* Medical Physics 35(1), 2008.
- A. Z. Zhu, L. Yuan, K. Chaney, K. Daniilidis. *Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion.* CVPR 2019.
- J. Hagenaars, F. Paredes-Vallés, G. de Croon. *Self-Supervised Learning of Event-Based Optical Flow with Spiking Neural Networks.* NeurIPS 2021.
- F. Paredes-Vallés, K. Y. W. Scheper, C. De Wagter, G. de Croon. *Taming Contrast Maximization for Learning Sequential, Low-latency, Event-based Optical Flow.* ICCV 2023.
- S. Shiba, Y. Aoki, G. Gallego. *Event Collapse in Contrast Maximization Frameworks.* Sensors 2022.
- S. Shiba, Y. Aoki, G. Gallego. *Secrets of Event-Based Optical Flow.* ECCV 2022.
- S. Meister, J. Hur, S. Roth. *UnFlow: Unsupervised Learning of Optical Flow with a Bidirectional Census Loss.* AAAI 2018.
- L. Liu et al. *Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation.* CVPR 2020.（ARFlow）
- C. Godard, O. Mac Aodha, M. Firman, G. Brostow. *Digging Into Self-Supervised Monocular Depth Estimation.* ICCV 2019.（Monodepth2）
- S. Tulyakov et al. *Time Lens++: Event-based Frame Interpolation with Parametric Non-linear Flow and Multi-scale Fusion.* CVPR 2022.（同样用参数化非线性运动，可作对比）

---

## 附录 A：默认超参数

| 位置 | 参数 | 默认值 |
| --- | --- | --- |
| 估计器 | K / 迭代次数（训练、推理） | 4 / 6、8–12 |
| 估计器 | 事件时间窗 S / 每窗 bin 数 / 目标窗 | 9 / 16 / (2, 4, 6, 8) |
| 估计器 | 相关金字塔层数 / 半径 | 4 / 3 |
| 估计器 | 循环残差时刻 | (0, 0.25, 0.5, 0.75) |
| 估计器 | 通道：特征 / 隐状态 / 上下文 | 128 / 128 / 128 |
| 估计器 | metric 限幅 | ±3 |
| 反转 | 不动点迭代次数 / splat 覆盖率阈值 / 分母下限 | 4 / 0.5 / 0.02 |
| 损失 | w_ph / w_end / w_cyc / w_sm / w_cm | 0.5 / 0.2 / 0.02 / 0.05 / 0 |
| 损失 | 序列权重 γ / 平滑边缘系数 α | 0.8 / 10 |
| 损失 | 前后向遮挡阈值 α₁ / α₂（全分辨率；1/4 分辨率用 α₂/16） | 0.01 / 0.5 |
| 损失 | CM：bin 步长 / 不动点次数 / 覆盖下限 | 4 / 4 / 0.05 |
| 训练 | lr / eta_min / 梯度裁剪 / warmup | 1e-4 / 1e-6 / 0.1 / 2000 步 |

## 附录 B：常见错误清单

- [ ] 没开 `--fix_warp`：`DACTModel` 会直接报错；如果你绕过了这个检查，BiW 和轨迹模块会差半个像素。
- [ ] 输入尺寸不是 8 的倍数：评测脚本的补边要从 4 改成 8。
- [ ] 锚点 1 的 voxel 只 flip 没取负：时间反转后极性也要反号。
- [ ] 锚点 1 查询时用了 s = τ：应该是 s = 1 − τ（只在 `TrajQuery` 和损失里换算）。
- [ ] 在 `gts.view(b*t, ...)` 之后才调 `TrajLoss`：形状不对。
- [ ] CM 损失用了归一化后的 voxel：要用原始 voxel。
- [ ] 用了 softsplat 的 `"avg"` / `"soft"` 模式做反转：会重新引入 `q/(q+1e-7)` 的梯度尖峰。
- [ ] 改了 `CorrBlock` 的窗口排列却没改 `AltCorrBlock`：训练和推理的通道对不上，`test_corr_full_equals_alt` 会失败。
- [ ] 评测时的 `--fix_*` 开关和训练时不一致：checkpoint 标志检查会报错，不要用 `--allow_config_mismatch` 绕过。