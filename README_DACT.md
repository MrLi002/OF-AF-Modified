# OF-AF + DACT (Dual-Anchor Continuous Trajectory)
Based on MrLi002/OF-AF-Modified commit b1d3741794c4a7b001d80f7fc24f7bf9b5c42e31.
This branch contains all upstream OF-AF files, previous OF-AF fixes and integrated DACT source.
The upstream branch remains unchanged.

## Start
Install OF-AF requirements, CUDA and CuPy softsplat dependencies, then run:
```bash
python -m pytest -q tests/test_ofaf_fixes.py tests/test_traj.py
bash train_traj.sh --gopro_root /path/to/GOPRO/train --save_dir train/dact
python bsergb_eval.py --checkpoint train/dact/epoch19.pth --test_root /path/to/1_TEST --fix_voxel --fix_warp --lpips_normalize --corr_impl alt --skip_flolpips
```

The DACT model is selected via --flow_mode traj and legacy OF-AF via --flow_mode bifeb.
For 24GB GPUs the default training script selects four interpolation targets per batch.
This is an experimental implementation: CPU components were tested previously, but full CUDA/CuPy training and metric improvement have NOT been validated here.
The reference design and known limitations are in docs/.
