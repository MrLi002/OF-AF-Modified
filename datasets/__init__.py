import os

from torch.utils.data import ConcatDataset
from .gopro import GOPRO_Train
from .bsergb import BSERGB_Train

def get_dataset(args):
    name = args.dataset
    datasets = []
    gopro_root = args.gopro_root
    bsergb_root = args.bsergb_root
    # root=r"D:\Users\cxc\Desktop\LJH\dataset\GOPRO_Large_all8\train"
    # root=r"D:\Users\cxc\Desktop\LJH\dataset\bsergb\3_TRAINING"
    if name == 'gopro':
        for scene in sorted(os.listdir(gopro_root)):
            if os.path.isdir(os.path.join(gopro_root, scene)):
                datasets.append(GOPRO_Train(args.voxel_bins, gopro_root, scene, 31,
                                           fix_voxel=args.fix_voxel))
    if name == 'bsergb':
        for skip in [1, 3]:
            for scene in sorted(os.listdir(bsergb_root)):
                if os.path.isdir(os.path.join(bsergb_root, scene)):
                    datasets.append(BSERGB_Train(args.voxel_bins, bsergb_root, scene, skip,
                                                 fix_voxel=args.fix_voxel))
    if not datasets:
        raise ValueError(f"No scenes found for {name}; check dataset root paths")
    if len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)