import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import glob
import cv2
import time

from basicsr.metrics.psnr_ssim import calculate_psnr
from basicsr.metrics.psnr_ssim import calculate_ssim
from basicsr.models.archs.mpftnet_arch import MPFTNet
import scipy.io as sio
from pdb import set_trace as stx
import sys
sys.path.append('../')

parser = argparse.ArgumentParser(description='Real Image Denoising using MIRNet_v2')

parser.add_argument('--input_dir', default='Real_Denoising/Datasets/test_crops/input_crops/', type=str, help='Directory of validation images')
parser.add_argument('--gt_dir', default='Real_Denoising/Datasets/test_crops/target_crops/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='Real_Denoising/MFPTNet_results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='experiments/RealDenoising_MPFTNet/models/net_g_8000.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

def save_to_file(pth, contents):
    fh = open(pth, 'a')
    fh.write(contents)
    fh.close()

####### Load yaml #######
yaml_file = 'Real_Denoising/Options/RealDenoising_MPFTNet.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################
os.makedirs(args.result_dir, exist_ok=True)

model_restoration = MPFTNet(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


img_paths = glob.glob(os.path.join(args.input_dir, '*.bmp'))
img_paths.sort()
img_paths_gt = glob.glob(os.path.join(args.gt_dir, '*.bmp'))
img_paths_gt.sort()


psnrs = []
ssims = []
tt = []
save_dict = {}
psnr_dict = {}
ssim_dict = {}

with torch.no_grad():
    for i in range(len(img_paths)):
        img_name = img_paths[i].split('/')[-1]

        img = cv2.imread(img_paths[i], -1).astype(np.float32)
        gt = cv2.imread(img_paths_gt[i], 0)
        img /=255.

        img = np.expand_dims(img, axis=-1)
        noisy_patch = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).cuda()
        start_time = time.time()
        restored_patch = model_restoration(noisy_patch)
        end_time = time.time() - start_time

        tt.append(end_time)
        restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
        restored_patch = restored_patch.numpy()
        gt_cuda = torch.from_numpy((gt/255.).astype(np.float32)).cuda()
        restored_patch_cuda = torch.from_numpy(restored_patch[:, :, 0]).cuda()
        restored_patch = np.clip(restored_patch * 255.0, 0, 255).astype(np.uint8)

        save_file = os.path.join(args.result_dir, img_name)
        utils.save_img(save_file, restored_patch)
        psnr_x_ = calculate_psnr(restored_patch[:, :, 0], gt, 0)
        ssim_x_ = calculate_ssim(restored_patch[:, :, 0], gt, 0)
        

        psnr_dict[img_name] = psnr_x_
        ssim_dict[img_name] = ssim_x_

        psnrs.append(psnr_x_)
        ssims.append(ssim_x_)

    psnr_avg = np.mean(psnrs)
    ssim_avg = np.mean(ssims)
    psnr_max = np.max(psnrs)
    ssim_max = np.max(ssims)

    txtname = os.path.join(args.result_dir, 'test_result.txt')
    if not os.path.exists(txtname):
        os.system(r"touch {}".format(txtname))

    save_to_file(os.path.join(args.result_dir, 'test_result.txt'),
                 "psnr_avg: {:.4f},  ssim_avg: {:.4f} \npsnr_max: {:.4f}, ssim_max:{:.4f}\n". \
                 format(psnr_avg, ssim_avg, psnr_max, ssim_max))


    p_str = " ".join(sys.argv)
    save_to_file(os.path.join(args.result_dir, 'test_result.txt'), p_str + '\n')

    # Save PSNR and SSIM for each image
    for k1, k2 in zip(psnr_dict, ssim_dict):
        save_to_file(os.path.join(args.result_dir, 'test_result.txt'),
                     "{}, psnr: {}, ssim: {}\n".format(k1, psnr_dict[k1], ssim_dict[k2]))
    print("Psnr_Avg:{:.5f}, ssim_avg:{:.5f} \nPsnr_Max: {:.5f}, ssim_max:{:.8f}".\
          format(psnr_avg, ssim_avg, psnr_max, ssim_max))
    
    print(np.mean(tt))