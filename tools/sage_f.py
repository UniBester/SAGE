import os 
import cv2
import numpy as np
import torch
from tqdm import tqdm
import argparse



def freq_conpensate(rimg, iimg, fimg, sigma=5):
    def lowpass(rimg, iimg, fimg, sigma, gamma1=0.8, gamma2=0.3):
        height, width = rimg.shape
    
        rfft = np.fft.fft2(rimg)
        rfft = np.fft.fftshift(rfft)

        ifft = np.fft.fft2(iimg)
        ifft = np.fft.fftshift(ifft)

        ffft = np.fft.fft2(fimg)
        ffft = np.fft.fftshift(ffft)
    
        for i in range(height):
            for j in range(width):
                rfft[i, j] *= np.exp(-((i - (height - 1)/2)**2 + (j - (width - 1)/2)**2)/2/sigma**2)
                ifft[i, j] *= np.exp(-((i - (height - 1)/2)**2 + (j - (width - 1)/2)**2)/2/sigma**2)


        fft=rfft-gamma2*ifft+gamma1*ffft
        fft = np.fft.ifftshift(fft)
        nfft = np.fft.ifft2(fft)
    
        nfft = np.real(nfft)
        max = np.max(nfft)
        min = np.min(nfft)
    
        res = np.zeros((height, width), dtype = "uint8")
    
        for i in range(height):
            for j in range(width):
                res[i, j] = 255 * (nfft[i, j] - min)/(max - min)
        return res

    def highpass(rimg, iimg, fimg, sigma, gamma1=0.8, gamma2=0.3):
        height, width = rimg.shape
    
        rfft = np.fft.fft2(rimg)
        rfft = np.fft.fftshift(rfft)

        ifft = np.fft.fft2(iimg)
        ifft = np.fft.fftshift(ifft)

        ffft = np.fft.fft2(fimg)
        ffft = np.fft.fftshift(ffft)
    
        for i in range(height):
            for j in range(width):
                rfft[i, j] *= (1 - np.exp(-((i - (height - 1)/2)**2 + (j - (width - 1)/2)**2)/2/sigma**2))
                ifft[i, j] *= (1 - np.exp(-((i - (height - 1)/2)**2 + (j - (width - 1)/2)**2)/2/sigma**2))


        fft=rfft-gamma2*ifft+gamma1*ffft
        fft = np.fft.ifftshift(fft)
        nfft = np.fft.ifft2(fft)
    
        nfft = np.real(nfft)
        max = np.max(nfft)
        min = np.min(nfft)
    
        res = np.zeros((height, width), dtype = "uint8")
    
        for i in range(height):
            for j in range(width):
                res[i, j] = 255 * (nfft[i, j] - min)/(max - min)
        return res
        
    bgr=[]
    for i in range(3):
        bgr.append(highpass(rimg[:,:,i], iimg[:,:,i], fimg[:,:,i], sigma))
    bgr_high=np.stack(bgr, axis=2)
    # cv2.imwrite('freq_decompose/high_gaussian.jpg', bgr_high)

    bgr=[]
    for i in range(3):
        bgr.append(lowpass(rimg[:,:,i], iimg[:,:,i], fimg[:,:,i], sigma))
    bgr_low=np.stack(bgr, axis=2)
    # cv2.imwrite('freq_decompose/low_gaussian.jpg', bgr_low)
    return bgr_high, bgr_low

if __name__=='__main__':
    


    parser = argparse.ArgumentParser(description='Metrics')
    parser.add_argument('real_data_path', type=str)
    parser.add_argument('inversion_path', type=int)
    parser.add_argument('edited_path', type=str)
    parser.add_argument('output_path', type=int)
    args = parser.parse_args()  
    cates = os.listdir(args.real_data_path)
    for cate in tqdm(cates):
        
        os.makedirs(os.path.join(args.output_path, 'freq_conpensate_low',cate),exist_ok=True)
        os.makedirs(os.path.join(args.output_path, 'freq_conpensate_high',cate),exist_ok=True)
        files = os.listdir(os.path.join(args.real_data_path, cate))
        for i in range(len(os.listdir(os.path.join(args.edited_path, cate)))):
            real_file = os.path.join(args.real_data_path, cate, files[i % len(files)])
            # print(real_file)
            inversion_file = os.path.join(args.inversion_path, cate, files[i % len(files)].split('.')[0]+'.jpg')
            edited_file = os.path.join(args.edited_path, cate, str(i)+'.jpg')
            
            try:
                real = cv2.imread(real_file)
                real = cv2.resize(real, [256, 256])
                inversion = cv2.imread(inversion_file)
                inversion = cv2.resize(inversion, [256, 256])
                edited = cv2.imread(edited_file)
                edited = cv2.resize(edited, [256, 256])
            except:
                print(real_file)
                continue
            

            bgr_high, bgr_low=freq_conpensate(real, inversion, edited, sigma=5)
            cv2.imwrite(os.path.join(args.output_path, 'freq_conpensate_low',cate,str(i)+'.jpg'), bgr_low)
            cv2.imwrite(os.path.join(args.output_path, 'freq_conpensate_high',cate,str(i)+'.jpg'), bgr_high)


