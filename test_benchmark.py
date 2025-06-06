import argparse
import os
from math import log10
import lpips
import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name

results = {
    'Set5': {'psnr': [], 'ssim': [], 'lpips': []}, 
    'Set14': {'psnr': [], 'ssim': [], 'lpips': []}, 
    'BSD100': {'psnr': [], 'ssim': [], 'lpips': []},
    'Urban100': {'psnr': [], 'ssim': [], 'lpips': []}, 
    'SunHays80': {'psnr': [], 'ssim': [], 'lpips': []}
          }

model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=32, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
    
# 初始化 lpips_loss
lpips_loss = lpips.LPIPS(net='vgg')
if torch.cuda.is_available():
    lpips_loss = lpips_loss.cuda()
    
with torch.no_grad():  # 使用torch.no_grad()上下文管理器
    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).mean().item()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()  # 使用item()方法
        lpips_score = lpips_loss(sr_image, hr_image).mean().item()  # 计算LPIPS分数

        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.cpu().squeeze(0)),
             display_transform()(sr_image.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) +
                         image_name.split('.')[-1], padding=5)

        # save psnr\ssim
        results[image_name.split('_')[0]]['psnr'].append(psnr)
        results[image_name.split('_')[0]]['ssim'].append(ssim)
        results[image_name.split('_')[0]]['lpips'].append(lpips_score)

out_path = 'statistics/'
saved_results = {'psnr': [], 'ssim': [], 'lpips': []}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    lpips = np.array(item['lpips'])
    if (len(psnr) == 0) or (len(ssim) == 0):
        psnr = 'No data'
        ssim = 'No data'
        lpips = 'No data'
    else:
        psnr = psnr.mean()
        ssim = ssim.mean()
        lpips = lpips.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)
    saved_results['lpips'].append(lpips)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')
