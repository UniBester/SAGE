import os
from tqdm import tqdm
import numpy as np
import sys
import argparse

import lpips
import cv2
import random
from PIL import Image


def fid(real, fake, gpu):
    print('Calculating FID...')
    print('real dir: {}'.format(real))
    print('fake dir: {}'.format(fake))
    command = 'python -m pytorch_fid {} {} --device cuda:{}'.format(real, fake, gpu)
    print(command)
    os.system(command)

def LPIPS(root):
    print('Calculating LPIPS...')
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    model = loss_fn_vgg
    model.cuda()

    files = os.listdir(root)
    data = {}
    for file in tqdm(files, desc='loading data'):
        cls = file.split('_')[0]
        idx = int(file.split('_')[1][:-4])
        img = lpips.im2tensor(cv2.resize(lpips.load_image(os.path.join(root, file)), (32, 32)))
        data.setdefault(cls, {})[idx] = img

    classes = set([file.split('_')[0] for file in files])
    res = []
    for cls in tqdm(classes):
        temp = []
        files_cls = [file for file in files if file.startswith(cls + '_')]
        for i in range(0, len(files_cls) - 1, 1):
            for j in range(i + 1, len(files_cls), 1):
                img1 = data[cls][i].cuda()
                img2 = data[cls][j].cuda()

                d = model(img1, img2, normalize=True)
                temp.append(d.detach().cpu().numpy())
        res.append(np.mean(temp))
    print(np.mean(res))




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Metrics')
    parser.add_argument('groundtruth_data_path', type=str, help='Path to the groundtruth data')
    parser.add_argument('generated_data_path', type=int, help='Path to the generated data')
    args = parser.parse_args()  
    
    # calculate metrics
    fid(args.groundtruth_data_path, args.generated_data_path, 0)
    LPIPS(args.generated_data_path)




