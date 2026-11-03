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
