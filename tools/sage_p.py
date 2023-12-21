import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse


def diff_mask(img1, img2):
    diff = np.abs(img1-img2)
    diff = np.sum(diff, axis=2)
    max = np.max(diff)
    min = np.min(diff)
    height, width = diff.shape
    mask = np.ones_like(diff).astype(np.float16)
    for i in range(height):
        for j in range(width):
            mask[i, j] = (diff[i, j] - min)/(max - min)
    return mask


def combine(real, inversion, edited, beta=0.5):
    ri_diff = diff_mask(real, inversion)
    ie_diff = diff_mask(inversion, edited)
    mask = ri_diff-beta*ie_diff
    height, width = mask.shape
    for i in range(height):
        for j in range(width):
            if mask[i, j] < 0:
                mask[i, j] = 0

    def gaussian_filter(img, K_size=3, sigma=1.3):

        if len(img.shape) == 3:
            H, W, C = img.shape
        else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape

        # Zero padding
        pad = K_size // 2
        out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
        out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

        # prepare Kernel
        K = np.zeros((K_size, K_size), dtype=np.float)
        for x in range(-pad, -pad + K_size):
            for y in range(-pad, -pad + K_size):
                K[y + pad, x +
                    pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
        K /= (2 * np.pi * sigma * sigma)
        K /= K.sum()
        tmp = out.copy()

        # filtering
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    out[pad + y, pad + x,
                        c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
        out = np.clip(out, 0, 1)
        out = out[pad: pad + H, pad: pad + W]
        return out

    mask = gaussian_filter(mask, K_size=3, sigma=1.3)
    img = real*mask+edited*(1-mask)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metrics')
    parser.add_argument('real_data_path', type=str)
    parser.add_argument('inversion_path', type=int)
    parser.add_argument('edited_path', type=str)
    parser.add_argument('output_path', type=int)
    args = parser.parse_args()  
    cates = os.listdir(args.real_data_path)
    file_list=np.load('nabirds/nabirds_file_list.npy', allow_pickle=True).item()
    for cate in tqdm(cates):
        os.makedirs(os.path.join(args.output_path, cate), exist_ok=True)
        files = os.listdir(os.path.join(args.real_data_path, cate))
        for i in range(len(os.listdir(os.path.join(args.edited_path, cate)))):
            real_file = os.path.join(args.real_data_path, cate, files[i % len(files)])
            inversion_file = os.path.join(args.inversion_path, cate, files[i % len(files)])
            edited_file = os.path.join(args.edited_path, cate, files[i % len(files)])
            
            try:
                real = cv2.imread(real_file)
                real = cv2.resize(real, [256, 256])
                inversion = cv2.imread(inversion_file)
                inversion = cv2.resize(inversion, [256, 256])
                edited = cv2.imread(edited_file)
                edited = cv2.resize(edited, [256, 256])
                img=combine(real, inversion, edited)
                cv2.imwrite(os.path.join(args.output_path, cate, str(i)+'.jpg'), img)
            except:
                print(real_file)
                continue
            
