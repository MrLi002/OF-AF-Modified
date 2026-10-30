import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
import cv2
import torch
import numpy as np

from tqdm import tqdm
from torch.nn import functional as F
from lpips import LPIPS
from flolpipsloss.flolpips import Flolpips
from models.model import MyModel
from util.event import EventSequence
from util.voxelization import to_voxel_grid
from util.alignment import natural_sort
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

class BSERGB_Test():
    def __init__(self, voxel_bins, skips=[1, 3], save_root='eval/bsergb',
                 save_pred=False, test_root='/DATASSD1/BSERGB/1_TEST',
                 fix_voxel=False, std_stride=False, lpips_normalize=False,
                 skip_flolpips=False):
        self.skips = skips
        self.voxel_bins = voxel_bins
        self.save_root = save_root
        self.save_pred = save_pred
        self.fix_voxel = fix_voxel
        self.std_stride = std_stride
        self.lpips_normalize = lpips_normalize
        self.skip_flolpips = skip_flolpips
        self.skip_inds = {
            'basket_09':[31, 32, 33, 34],
            'may29_rooftop_handheld_02':[17, 70],
            'may29_rooftop_handheld_03':[306],
            'may29_rooftop_handheld_05':[21],
        }
        self.test_root = test_root
        self.all_scenes = ['acquarium_08', 'ball_05', 'ball_06', 'basket_08', 'basket_09', 'candies_03', 'eggs_04', 'elastic_bands_01', 'fire_02', 
                           'football_04', 'horse_11', 'horse_12', 'horse_13', 'horse_18', 'horse_20', 'jacket_03', 'juggling_06', 'may28_axe_01', 
                           'may29_redbull_01', 'may29_water_tank_pouring_02', 'orange_juice_02', 'paprika_1000_gain_control_02', 'pen_03', 
                           'rope_jumping_01', 'street_crossing_08', 'watermelon_01']
        self.test_scenes = self.all_scenes
        self.lpips = LPIPS(net="alex").cuda()
        self.flolpips = None if skip_flolpips else Flolpips().cuda()

    def init_scene(self, scene, skip):
        self.cur_skip = skip
        self.cur_scene = scene
        self.skip_dir = os.path.join(self.save_root, str(skip))
        self.save_dir = os.path.join(self.skip_dir, scene)
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, 'log.txt')

        self.img_folder = os.path.join(self.test_root, scene, 'images')
        self.img_names = os.listdir(self.img_folder)
        self.img_names = [file_name for file_name in self.img_names if file_name.endswith(".png")]
        self.img_names = natural_sort(self.img_names) if self.fix_voxel else sorted(self.img_names)
        self.h, self.w, _ = cv2.imread(os.path.join(self.img_folder, self.img_names[0])).shape

        self.event_folder = os.path.join(self.test_root, scene, 'events')
        self.event_names = [f for f in os.listdir(self.event_folder) if f.endswith(".npz")]
        self.event_names = natural_sort(self.event_names) if self.fix_voxel else sorted(self.event_names)
        if self.fix_voxel and len(self.event_names) != len(self.img_names) - 1:
            raise ValueError(f"{scene}: expected images-1 event interval files")

        self.cur_skip_inds = []
        if scene in self.skip_inds:
            self.cur_skip_inds = self.skip_inds[scene]

    def get_img(self, index):
        ind = list(range(index, index+self.cur_skip+2))
        imgs = torch.cat([torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(self.img_folder, self.img_names[i])), cv2.COLOR_BGR2RGB)) for i in ind], dim=2)
        return imgs, ind

    def get_events(self, ind, size=None):
        events_path = [os.path.join(self.event_folder, i) for i in self.event_names[ind[0]:ind[-1]]]
        loader = (EventSequence.from_frame_interval_files if self.fix_voxel
                  else EventSequence.from_npz_files)
        events_01 = loader(events_path, self.h, self.w, bsergb=True, size=size)
        return events_01

    def get_batch(self, index):
        imgs, ind = self.get_img(index)
        self.h, self.w, c = imgs.shape
        events_01 = self.get_events(ind)
        voxel_01 = (to_voxel_grid(events_01, self.voxel_bins) if events_01 is not None
                    else torch.zeros(self.voxel_bins, self.h, self.w))
        imgs = imgs.permute(2, 0, 1)/255
        times = (torch.tensor(ind[1:-1])-ind[0])/(self.cur_skip+1)

        imgs = imgs.unsqueeze(0)
        voxel_01 = voxel_01.unsqueeze(0)
        times = times.unsqueeze(0)
        return imgs, voxel_01, times

    def pad(self, tensor):
        b, c, h, w = tensor.shape
        h_pad = (4 - h % 4) % 4
        w_pad = (4 - w % 4) % 4
        tensor_pad = F.pad(tensor, [0, w_pad, 0, h_pad], mode="constant", value=0)
        return tensor_pad

    def cal_psnr(self, gt, pred):
        return peak_signal_noise_ratio(gt, pred, data_range=255)

    def cal_ssim(self, gt, pred):
        return structural_similarity(gt, pred, data_range=255, channel_axis=2)

    def cal_flolpips(self, img0, img1, pred, gt):
        flolpips = self.flolpips.forward(img0, img1, pred, gt)
        return flolpips.item()

    def cal_lpips(self, gt, pred):
        return self.lpips(gt, pred, normalize=self.lpips_normalize).item()

    def write_log(self, log, path):
        with open(path, 'a') as f:
            f.write(log + '\n')

    def test(self, model):
        model.eval()
        with torch.no_grad():
            for skip in self.skips:
                all_psnr = []
                all_ssim = []
                all_flolpips = []
                all_lpips = []
                for scene in self.test_scenes:
                    self.init_scene(scene, skip)
                    psnrs = []
                    ssims = []
                    flolpipss = []
                    lpipss = []
                    for index in tqdm(range(0, len(self.img_names)-self.cur_skip-1,
                                             skip + 1 if self.std_stride else skip), desc='Current: Test on {}_{}'.format(scene, skip)):
                        tar = 0
                        for ind in range(index, index+skip+2):
                            if ind in self.cur_skip_inds:
                                tar = 1
                        if tar == 1:
                            print("skip {}/{}".format(scene, index))
                            continue
                        
                        imgs, voxel01, times = self.get_batch(index)
                        b, c, h, w = imgs.shape
                        imgs = self.pad(imgs)
                        voxel01 = self.pad(voxel01)

                        imgs = imgs.cuda()
                        im0 = imgs[:, :3]
                        gts = imgs[:, 3:-3]
                        im1 = imgs[:, -3:]
                        voxel01 = voxel01.cuda()
                        times = times.cuda()
                        # res = self.batch_eval(model, im0, im1, voxel01, times, skip)
                        res = model(im0, im1, voxel01, times)

                        preds = torch.clamp(res, 0, 1)[:, :, :, :h, :w]
                        gts = torch.stack(gts.split(3, dim=1), dim=1)[:, :, :, :h, :w] #1, t, c, h, w
                        im0 = im0[:, :, :h, :w]
                        im1 = im1[:, :, :h, :w]
                        for i in range(skip):
                            flolpips = (self.cal_flolpips(im0, im1, preds[:, i], gts[:, i])
                                         if self.flolpips is not None else float("nan"))
                            lpips = self.cal_lpips(preds[:, i], gts[:, i])
                            pred = (preds[0, i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                            gt = (gts[0, i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                            psnr = self.cal_psnr(gt, pred)
                            ssim = self.cal_ssim(gt, pred)

                            log = "[{}_{}/{}]\tPSNR:{:.2f}\tSSIM:{:.4f}\tLPIPS:{:.4f}\tFloLPIPS:{:.4f}".format(scene, skip, self.img_names[index + i + 1], psnr, ssim, lpips, flolpips)
                            self.write_log(log, self.log_path)
                            if self.save_pred:
                                save_path = os.path.join(self.save_dir, self.img_names[index + i + 1])
                                cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                            psnrs.append(psnr)
                            ssims.append(ssim)
                            flolpipss.append(flolpips)
                            lpipss.append(lpips)
                            all_psnr.append(psnr)
                            all_ssim.append(ssim)
                            all_flolpips.append(flolpips)
                            all_lpips.append(lpips)
                    log = "[{}_{}]\tMean PSNR:{:.2f}\tMean SSIM:{:.4f}\tMean LPIPS:{:.4f}\tMean FloLPIPS:{:.4f}".format(scene, skip, np.mean(psnrs), np.mean(ssims), np.mean(lpipss), np.mean(flolpipss))
                    print(log)
                    self.write_log(log, self.log_path)
                    self.write_log(log, os.path.join(self.skip_dir, 'bsergb.txt'))
                log = "[bsergb_{}]\tMean PSNR:{:.2f}\tMean SSIM:{:.4f}\tMean LPIPS:{:.4f}\tMean FloLPIPS:{:.4f}".format(skip, np.mean(all_psnr), np.mean(all_ssim), np.mean(all_lpips), np.mean(all_flolpips))
                print(log)
                self.write_log(log, os.path.join(self.skip_dir, 'bsergb.txt'))
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='train/bsergb.pth', type=str, help='checkpoint path')
    parser.add_argument('--save_pred', action='store_true')
    parser.add_argument('--nb_of_flow', default=16, type=int)
    parser.add_argument('--voxel_bins', default=128, type=int)
    parser.add_argument('--test_root', default='/DATASSD1/BSERGB/1_TEST')
    parser.add_argument('--fix_query', action='store_true')
    parser.add_argument('--fix_warp', action='store_true')
    parser.add_argument('--fix_voxel', action='store_true')
    parser.add_argument('--std_stride', action='store_true',
                        help='Use skip+1 rather than legacy skip; changes evaluation protocol')
    parser.add_argument('--lpips_normalize', action='store_true',
                        help='Correct [-1,1] LPIPS inputs; legacy scores are not comparable')
    parser.add_argument('--skip_flolpips', action='store_true',
                        help='Run PSNR/SSIM/LPIPS without external FloLPIPS weights')
    parser.add_argument('--allow_config_mismatch', action='store_true')
    args = parser.parse_args()

    model = MyModel(args)
    state = torch.load(args.checkpoint, map_location='cpu')
    flags = state.get("preprocessing_flags")
    current = {k: bool(getattr(args, k)) for k in ("fix_query", "fix_warp", "fix_voxel")}
    if flags is not None and any(flags.get(k) != value for k, value in current.items()):
        if not args.allow_config_mismatch:
            raise ValueError(f"Checkpoint trained with {flags}, evaluation requested {current}; "
                             "pass matching --fix_* flags, or --allow_config_mismatch intentionally")
        print("WARNING: evaluating with mismatched fix flags.")
    elif flags is None:
        print("NOTE: legacy checkpoint contains no recorded fix flags.")
    cpk = state['model']
    new_state_dict = {k.replace('module.', ''): v for k, v in cpk.items()}
    model.load_state_dict(new_state_dict)
    model.cuda()

    print(f'Loaded checkpoint from {args.checkpoint}')
    save_root = "eval/bsergb"

    test = BSERGB_Test(args.voxel_bins, [3, 1], save_root,
                      save_pred=args.save_pred, test_root=args.test_root,
                      fix_voxel=args.fix_voxel, std_stride=args.std_stride,
                      lpips_normalize=args.lpips_normalize,
                      skip_flolpips=args.skip_flolpips)
    test.test(model)
