# OF-AF 代码替换说明（基于提交 b1d3741）

适用于分支 `fix/ofaf-reproducibility-20261030` 的提交 **b1d3741**。改完的效果和补丁 `ofaf_b1d3741_followup.patch` 完全相同，两种方式任选一种。

- 文中行号都按 b1d3741 计算。改动多处时，前面的改动会让后面的行号往下错几行，所以请以“原代码”为准，用编辑器搜索定位。
- `models/model.py` 用的是 **Tab 缩进**，其余 Python 文件是 4 个空格。替换时保持原文件的缩进方式，不要混用。

## 改动总览

| # | 要解决的问题 | 修改的文件 |
| --- | --- | --- |
| 1 | 空洞掩码 `q/(q+1e-7)` 对光流的梯度会出现尖峰（最高约 1e5 倍） | `models/BiW.py`、`models/model.py`、`train_model.py`、`train_model.sh`、`finetune_bsergb.sh` |
| 2 | 事件文件比图像多得多时（例如 ×8 V2E 每个帧间隔 8 个文件），会被静默截断、错位配对 | `util/alignment.py`、`check_events.py` |
| 3 | BS-ERGB 时间戳读入时就转成 float32，大数值时精度不够 | `util/event.py` |
| 4 | 为第 2、3 条补测试 | `tests/test_ofaf_fixes.py` |

---

## 修改 1：空洞掩码的梯度尖峰

**原因：** avg 模式把覆盖率 q 归一化成 q/(q+1e-7)，它对光流的导数是 1e-7/(q+1e-7)²：几乎所有像素上约等于 0，在几乎没被覆盖的像素上可达 1e5。新增开关 `--fix_mask_grad`，打开后这条路径不再向光流传梯度。前向计算结果完全不变，所以评测官方权重不受影响；前向光流改由辅助损失（`--fwd_loss_weight`）单独监督。

### 1.1 `models/BiW.py` 第 23–25 行：`MaskGuideNet.__init__` 开头

原代码：

```python
class MaskGuideNet(nn.Module):
    def __init__(self, base_channel, fix_warp=False):
        super().__init__()
```

替换为：

```python
class MaskGuideNet(nn.Module):
    def __init__(self, base_channel, fix_warp=False, fix_mask_grad=False):
        super().__init__()
        self.fix_mask_grad = fix_mask_grad
```

### 1.2 `models/BiW.py` 第 44 行：生成 `warp_mask` 的那一行

原代码：

```python
        warp_mask = softsplat(ones, flow0t, tenMetric=None, strMode="avg")
```

替换为：

```python
        # avg 模式下覆盖率被归一化成 q/(q+1e-7)，它对光流的导数 1e-7/(q+1e-7)^2
        # 在几乎没被覆盖的像素上可达 1e5，会产生梯度尖峰。
        # fix_mask_grad=True 时这里不向光流传梯度（前向数值不变），前向光流只由辅助损失监督。
        warp_mask = softsplat(ones, flow0t.detach() if self.fix_mask_grad else flow0t,
                              tenMetric=None, strMode="avg")
```

### 1.3 `models/BiW.py` 第 75–78 行：`BiW.__init__` 开头

原代码：

```python
class BiW(nn.Module):
    def __init__(self, base_channel, fix_warp=False):
        super().__init__()
        self.fuse_warp = MaskGuideNet(base_channel, fix_warp=fix_warp)
```

替换为：

```python
class BiW(nn.Module):
    def __init__(self, base_channel, fix_warp=False, fix_mask_grad=False):
        super().__init__()
        self.fuse_warp = MaskGuideNet(base_channel, fix_warp=fix_warp, fix_mask_grad=fix_mask_grad)
```

### 1.4 `models/model.py`（注意 Tab 缩进）

**第 16 行之后新增一行。** 第 16 行原文是：

```python
		self.fix_warp = getattr(args, "fix_warp", False)
```

在它下面加：

```python
		self.fix_mask_grad = getattr(args, "fix_mask_grad", False)  # 只影响训练，推理结果不变
```

**第 24 行替换。** 原代码：

```python
		self.biw = BiW(self.base_channel, fix_warp=self.fix_warp)
```

替换为：

```python
		self.biw = BiW(self.base_channel, fix_warp=self.fix_warp, fix_mask_grad=self.fix_mask_grad)
```

### 1.5 `train_model.py`

**第 135 行之后新增命令行参数。** 第 135 行原文是：

```python
    parser.add_argument('--fix_voxel', action='store_true')
```

在它下面加：

```python
    parser.add_argument('--fix_mask_grad', action='store_true',
                        help='Stop the spiky flow gradient through the avg-normalised hole mask (training only)')
```

**第 114 行之后新增一行，把开关记进 checkpoint。** 第 114 行原文是：

```python
            'training_options': {'fwd_loss_weight': args.fwd_loss_weight,
```

在它下面加（缩进和下一行的 `'lpips_normalize'` 对齐）：

```python
                                 'fix_mask_grad': args.fix_mask_grad,
```

改完后这一段应该是：

```python
            'training_options': {'fwd_loss_weight': args.fwd_loss_weight,
                                 'fix_mask_grad': args.fix_mask_grad,
                                 'lpips_normalize': args.lpips_normalize},
```

### 1.6 `train_model.sh` 第 18 行、`finetune_bsergb.sh` 第 22 行

两个脚本里都有这一行：

```bash
    --fwd_loss_weight 0.1 \
```

在它下面各加一行：

```bash
    --fix_mask_grad \
```

行尾的反斜杠 `\` 不能漏，否则后面的参数不会传进去。

---

## 修改 2：事件文件数检查

**原因：** 现在的检查只要求事件文件“不少于图像数−1”，多出来的一律当作尾部文件截掉。如果你的数据是每个帧间隔有多个文件（例如 ×8 V2E 的输出），这样做会把事件和帧错位配对，而且只打印一个警告。改成：文件数只允许等于“图像数−1”或“图像数”，再多就报错。

### 2.1 `util/alignment.py` 第 27–34 行

原代码：

```python
    if event_count > required:
        warnings.warn(
            f"{scene}: ignoring {event_count - required} extra trailing event file(s); "
            "verify that event file 0 corresponds to image frame 0 -> 1",
            UserWarning,
            stacklevel=2,
        )
    return required
```

替换为：

```python
    if event_count > image_count:
        # 例如 ×8 V2E 每个帧间隔 8 个文件：直接截断会把事件和帧错位配对
        raise ValueError(
            f"{scene}: {event_count} event files for {image_count} images; expected images - 1 "
            "(one file per adjacent frame pair) or images (one trailing file). "
            "Several files per frame interval need a timestamp-based loader."
        )
    if event_count > required:
        warnings.warn(
            f"{scene}: ignoring 1 extra trailing event file; "
            "verify that event file 0 corresponds to image frame 0 -> 1",
            UserWarning,
            stacklevel=2,
        )
    return required
```

前面判断“少于图像数−1 就报错”的那段代码保持不变。

### 2.2 `check_events.py` 第 27–29 行

原代码：

```python
    elif len(events) > required:
        print(f"[warn] {len(events) - required} extra trailing event file(s) ignored "
              "(verify first-file index before training)")
```

替换为：

```python
    elif len(events) > len(images):
        bad.append(f"{len(events)} event files for {len(images)} images: more than one file per frame "
                   "interval (e.g. x8 V2E output); the interval loader would mis-pair them")
    elif len(events) > required:
        print("[warn] 1 extra trailing event file ignored (verify first-file index before training)")
```

它上面的 `if len(events) < required:` 那两行保持不变。

---

## 修改 3：BS-ERGB 时间戳保留 float64

**原因：** float32 的精度随数值变大而变差，在 1e9 附近只能精确到 64。改成 float64 后，后面的归一化流程会自然得到精确结果。

### `util/event.py` 第 62 行（`elif bsergb:` 分支里）

原代码：

```python
            tmp["timestamp"].astype(np.float32).reshape((-1,)),
```

替换为：

```python
            tmp["timestamp"].astype(np.float64).reshape((-1,)),  # float32 在大数值时精度不够
```

只改这一行。同一分支里的 `x`、`y`、`polarity` 保持 float32 不变。文件第 106 行附近还有一个被注释掉的旧版 `load_events`，不用管它。

---

## 修改 4：补充测试

### 4.1 `tests/test_ofaf_fixes.py` 第 94–95 行之后

第 94–95 行原文是：

```python
    with pytest.raises(ValueError, match="need at least images - 1"):
        validate_interval_file_count(4, 2, "scene")
```

在它下面加两行（缩进与上面相同）：

```python
    with pytest.raises(ValueError, match="more|expected images - 1"):
        validate_interval_file_count(4, 24, "scene")   # 例如每个帧间隔 8 个文件
```

### 4.2 `tests/test_ofaf_fixes.py` 文件末尾追加

```python


def test_bsergb_timestamps_keep_integer_resolution(tmp_path):
    path = tmp_path / "bsergb.npz"
    np.savez(path, x=np.array([0, 32, 64]), y=np.array([0, 32, 64]),
             timestamp=np.array([1000000000, 1000000001, 1000000002], dtype=np.int64),
             polarity=np.array([0, 1, 1]))
    events = load_events(path, bsergb=True)
    assert np.unique(events[:, 2]).size == 3
```

---

## 改完后检查

```bash
# 1. 单元测试：应全部通过（原有 10 个 + 新增 1 个，共 11 个）
python -m pytest -q tests/test_ofaf_fixes.py

# 2. 确认每处都改到了
grep -n "fix_mask_grad" models/BiW.py models/model.py train_model.py train_model.sh finetune_bsergb.sh
grep -n "event_count > image_count" util/alignment.py
grep -n 'timestamp"\].astype(np.float64)' util/event.py

# 3. 检查数据（GOPRO 和 BS-ERGB 各挑几个场景），输出里不能有 [!]
python check_events.py /path/to/GOPRO_Large_all/train/GOPR0372_07_00
python check_events.py /path/to/3_TRAINING/<场景名> --bsergb --skip 3

# 4. GPU 单场景冒烟测试
mkdir -p /tmp/gopro_one
ln -s /path/to/GOPRO_Large_all/train/GOPR0372_07_00 /tmp/gopro_one/
bash train_model.sh --gopro_root /tmp/gopro_one --epoch 1 --save_dir train/smoke
python check_ckpt.py train/smoke/epoch0.pth
```

冒烟测试的通过标准：

- 训练不报错。
- 进度条里的 `fwd_loss` 在下降。
- `check_ckpt.py` 输出的 `Forward / other median ratio` 在 1e-3 以上（官方权重是 4e-10）。
- 如果第 2 条的新检查报错，说明你的事件文件和帧不是一对一的，需要先确认数据布局，不要绕过这个检查。

## 使用注意

- **评测官方权重 `bsergb.pth`：** 照旧不加任何 `--fix_*` 开关。`--fix_mask_grad` 只在训练时起作用，评测不需要加。
- **修改 3 对旧结果的影响：** 只有 BS-ERGB 时间戳数值很大时，修改 3 才会让官方权重的评测结果有细微变化；时间戳在 1e7 以内时，结果和原来一致。