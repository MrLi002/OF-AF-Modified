# OF-AF reproducibility / experimental fixes

These modifications preserve the published-checkpoint inference behavior by default.
They **do not** prove an improvement in BS-ERGB PSNR until full GPU training/evaluation.
Read [check_events.py](check_events.py) before training on locally generated V2E files.

## Experimental flags

| Option | Default | Purpose |
|---|---|---|
| `--fix_query` | off | correct the reversed forward/right-flow interpolation and Query interpolation coefficients |
| `--fix_warp` | off | use proper pixel centers with `align_corners=False` |
| `--fix_voxel` | off | map each event *file* to a single frame interval (see caveat below) |
| `--fwd_loss_weight 0.1` | 0 | supervise forward-splatted images with masked Charbonnier; **not** in the paper |
| `--lpips_normalize` | off | convert LPIPS inputs from [0,1] to [-1,1], changing the original training loss / metric |
| `--std_stride` (evaluation) | off | use stride=skip+1; original paper evaluation stride=skip |
| `--skip_flolpips` (evaluation) | off | evaluate other metrics if FloLPIPS weight is missing |
| `--allow_config_mismatch` | off | explicit opt-out from stored checkpoint flag validation |

- `train_model.sh` and `finetune_bsergb.sh` are **fixed-model** recipes.
- `env.txt` includes the separate legacy recipe. The modified training defaults
  use learning rate 1e-4, final rate 1e-6; fine-tuning uses 3e-5 for 5 epochs.
- The official `bsergb.pth` is a **legacy** checkpoint: do **not** turn on
  `--fix_query` or `--fix_warp` when benchmarking it.
- Re-run training for altered geometry, event polarization, or LPIPS loss.
- Saved checkpoints include `preprocessing_flags`; train/eval reject mismatches.

### About event/frame synchronization

The input layout `images/*.png` + `events/*.npz` is assumed to contain one event
file for each adjacent pair of video frames, in matching order. `--fix_voxel`
maps file k to physical interval k/(skip+1)..(k+1)/(skip+1), but it estimates the
intra-file event positions using observed min/max event timestamps. It **cannot**
recover exact frame boundaries in sparse/eventless windows. Thus it is an
**approximation**, not a guaranteed timestamp synchronization solution.
For reproducibility, record real frame timestamps and event clock units when
available and extend preprocessing to use measured boundaries.

### Data and metrics

`load_events` now converts 0/1 polarity to -1/+1 in the generic GOPRO loader;
data previously trained with incorrect polarity should be retrained.

`BSERGB_Train` excludes the same known bad frame indices as the evaluation.
Legacy evaluation retains its original stride and uint8 truncation; they are
deliberately not silently changed. Turning on LPIPS normalization or standard
stride changes the benchmark definition.

FloLPIPS external `alex.pth` weights are **not redistributed** here. Obtain the
correct version through its upstream provider and place it at
`flolpipsloss/weights/v0.1/alex.pth`, or run `--skip_flolpips`.
The `.gitignore` exception only permits tracking the file; it does not
download it.

### How to check

```bash
python check_events.py "/path/to/GOPRO/train/scene"
python check_events.py "/path/to/3_TRAINING/scene" --bsergb --skip 3
pytest -q tests/test_ofaf_fixes.py
bash train_model.sh --gopro_root "/path/to/GOPRO/train"
bash finetune_bsergb.sh --bsergb_root "/path/to/3_TRAINING"
python bsergb_eval.py --checkpoint train/bsergb_ft/epoch4.pth --test_root "/path/to/1_TEST" --fix_query --fix_warp --fix_voxel --lpips_normalize
```

`check_ckpt.py` prints the Adam `exp_avg` moment magnitude of forward-flow
GRU and output heads; the published checkpoint's optimizer may not be available.

**Environment note:** The training forward path uses a CuPy CUDA softsplat
kernel, so CPU unit tests do not verify full end-to-end backpropagation.
Run a single GPU training batch and inspect the forward-flow parameter
gradients and the auxiliary loss before committing to an expensive full run.
