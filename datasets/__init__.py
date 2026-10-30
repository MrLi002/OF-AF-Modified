import os

from torch.utils.data import ConcatDataset
from .gopro import GOPRO_Train
from .bsergb import BSERGB_Train

def get_dataset(args):
    name = args.dataset
    datasets = []
    gopro_root=r"/home/yuaobo/桌面/ljh/dataset/GOPRO_Large_all/train"
    bsergb_root=r"/home/yuaobo/桌面/ljh/dataset/3_TRAINING"
    # root=r"D:\Users\cxc\Desktop\LJH\dataset\GOPRO_Large_all8\train"
    # root=r"D:\Users\cxc\Desktop\LJH\dataset\bsergb\3_TRAINING"
    if name == 'gopro':
        for scene in os.listdir(gopro_root):
            datasets.append(GOPRO_Train(args.voxel_bins, gopro_root, scene, 31))
    if name == 'bsergb':
        for skip in [1, 3]:
            for scene in os.listdir(bsergb_root):
                datasets.append(BSERGB_Train(args.voxel_bins, bsergb_root, scene, skip))
    if len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)