import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import os
from tqdm import tqdm
from pdb import set_trace as stx

tar = 'Real_Denoising/Datasets/train_crops'

lr_tar = os.path.join(tar, 'input_crops')
hr_tar = os.path.join(tar, 'target_crops')

os.makedirs(lr_tar, exist_ok=True)
os.makedirs(hr_tar, exist_ok=True)

lr_file = 'Real_Denoising/Datasets/train/input'
hr_file = 'Real_Denoising/Datasets/train/gt'
lr_files = glob(os.path.join(lr_file, '*bmp'))
lr_files.sort()
hr_files = glob(os.path.join(hr_file, '*bmp'))
hr_files.sort()

files = [(i, j) for i, j in zip(lr_files, hr_files)]

patch_size = 256
overlap = 64
p_max = 0

def save_files(file_):
    lr_file, hr_file = file_
    filename = os.path.splitext(os.path.split(lr_file)[-1])[0]
    lr_img = cv2.imread(lr_file)
    hr_img = cv2.imread(hr_file)
    num_patch = 0
    w, h = lr_img.shape[:2]
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-patch_size, patch_size-overlap, dtype=np.int32))
        h1 = list(np.arange(0, h-patch_size, patch_size-overlap, dtype=np.int32))
        w1.append(w-patch_size)
        h1.append(h-patch_size)
        for i in w1:
            for j in h1:
                num_patch += 1
                
                lr_patch = lr_img[i:i+patch_size, j:j+patch_size,:]
                hr_patch = hr_img[i:i+patch_size, j:j+patch_size,:]
                lr_patch = lr_patch[:, :, 0]
                hr_patch = hr_patch[:, :, 0]
                
                lr_savename = os.path.join(lr_tar, filename + '-' + str(num_patch) + '.bmp')
                hr_savename = os.path.join(hr_tar, filename + '-' + str(num_patch) + '.bmp')
                
                cv2.imwrite(lr_savename, lr_patch)
                cv2.imwrite(hr_savename, hr_patch)

    else:
        lr_savename = os.path.join(lr_tar, filename + '.bmp')
        hr_savename = os.path.join(hr_tar, filename + '.bmp')

        lr_img = lr_img[:, :, 0]
        hr_img = hr_img[:, :, 0]
        cv2.imwrite(lr_savename, lr_img)
        cv2.imwrite(hr_savename, hr_img)

for file_ in tqdm(files):
    save_files(file_)

