import os
from tqdm import tqdm
import numpy as np
import sys
from argparse import Namespace
import shutil

import torch
import torch.nn.functional as F
import lpips
import cv2
import random
from PIL import Image

sys.path.append(".")
sys.path.append("..")
os.environ['CUDA_VISIBLE_DEVICES']='0'

from options.test_options import TestOptions
from configs import data_configs
from utils.common import tensor2im
from options.test_options import TestOptions
from models.age import AGE

SIMILAR_CATES=-1
ALPHA=1
CM=1
OUTPUT_PATH='classifier/experiment/class_retention/nabirds/sage1.0q'
TEST_DATA_PATH='classifier/experiment/class_retention/nabirds/fewshot'
N_IMAGES=100

def get_ns(net, transform, class_embeddings, opts):
    samples=os.listdir(opts.train_data_path)
    ns_cate={}
    for s in tqdm(samples):
        cate=s.split('_')[0]
        if cate not in ns_cate.keys():
            ns_cate[cate]=[]
        ce=class_embeddings[cate].cuda()
        from_im = Image.open(os.path.join(opts.train_data_path,s))
        from_im = from_im.convert('RGB')
        from_im = transform(from_im)
        with torch.no_grad():
            latents = net.get_latents(from_im.unsqueeze(0).to("cuda").float())
            dw = latents[0][:6] - ce[:6]
            n = torch.linalg.lstsq(net.ax.A, dw).solution
            ns_cate[cate].append(n)
    os.makedirs(opts.n_distribution_path, exist_ok=True)
    torch.save(ns_cate, os.path.join(opts.n_distribution_path, 'ns.pt'))

def torchOrth(A, r=10):
    u,s,v = torch.svd(A)
    return v.T[:r]

def calc_statis(codes):
    codes=torch.stack(codes).permute(1,0,2).cpu().numpy()
    mean=np.mean(codes,axis=1)
    mean_abs=np.mean(np.abs(codes),axis=1)
    cov=[]
    for i in range(codes.shape[0]):
        cov.append(np.cov(codes[i].T))
    return {'mean':mean, 'mean_abs':mean_abs, 'cov':cov}

def get_similar_cate(class_embeddings, ce, k=20):
    keys=list(class_embeddings.keys())
    distances={}
    for key in keys:
        distances[key]=torch.sum(F.pairwise_distance(ce, class_embeddings[key].cuda(), p=2))
    cates=sorted(distances.items(), key=lambda x: x[1])[:k]
    cates=[i[0] for i in cates] 
    return cates

def get_local_distribution(latents, cr_directions, ns, class_embeddings, k=20):
    ce = get_ce(latents, cr_directions)
    cates = get_similar_cate(class_embeddings, ce, k)
    local_ns=[]
    for cate in cates:
        if cate not in ns.keys():
            print(cate)
            print('missed')
        else:
            local_ns+=ns[cate]
    return calc_statis(local_ns)

def get_crdirections(class_embeddings, r=30):
    class_embeddings=torch.stack(list(class_embeddings.values()))
    class_embeddings=class_embeddings.permute(1,0,2).cuda()
    cr_directions=[]
    for i in range(class_embeddings.shape[0]):
        cr_directions.append(torchOrth(class_embeddings[i], r))
    cr_directions=torch.stack(cr_directions)
    return cr_directions

def sampler(A, latents, dist, cr_dictionary, flag=True, alpha=1, l=50):
    ce=get_ce(latents[0], cr_dictionary).unsqueeze(0)
    means=dist['mean']
    covs=dist['cov']
    means_abs=torch.from_numpy(dist['mean_abs'])
    dws=[]
    for i in range(means.shape[0]):
        n=torch.from_numpy(np.random.multivariate_normal(mean=means[i], cov=covs[i], size=1)).float().cuda()
        #mask directions in A
        one = torch.ones_like(torch.from_numpy(means[0]))
        zero = torch.zeros_like(torch.from_numpy(means[0]))
        sorted, inds = torch.sort(means_abs[i], descending=True)
        beta = sorted[l]
        mask = torch.where(means_abs[i]>beta, one, zero).cuda()
        n=n*mask
        dw=torch.matmul(A[i], n.transpose(0,1)).squeeze(-1)
        dws.append(dw)
    dws=torch.stack(dws)
    if flag:
        codes = torch.cat(((alpha*dws.unsqueeze(0)+ ce[:, :6]), ce[:, 6:]), dim=1)
    else:
        codes = torch.cat(((alpha*dws.unsqueeze(0)+ latents[:, :6]), latents[:, 6:]), dim=1)
    return codes


def get_ce(latents, cr_directions):
    ce=[]
    for i in range(latents.shape[0]):
        cr_code=torch.zeros_like(latents[0])
        for j in range(cr_directions.shape[1]):
            cr_code=cr_code+torch.dot(latents[i],cr_directions[i][j])*cr_directions[i][j]
        ce.append(cr_code)
    ce=torch.stack(ce)
    return ce


if __name__=='__main__':
    SEED = 0
    print("========")
    random.seed(SEED)
    np.random.seed(SEED)

    #load model
    test_opts = TestOptions().parse()
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)
    net = AGE(opts)
    net.eval()
    net.cuda()
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    transform=transforms_dict['transform_inference']
    class_embeddings=torch.load(opts.class_embedding_path, map_location=torch.device("cpu"))
    cr_directions = get_crdirections(class_embeddings, r=10)
    cr_dictionary = get_crdirections(class_embeddings, r=-1)


    # get n distribution (only needs to be executed once)
    # get_n_distribution(net, transform, class_embeddings, test_opts)
    # get_ns(net, transform, class_embeddings, test_opts)
    # exit(0)



    # generate data
    ns = torch.load(os.path.join(opts.n_distribution_path, 'ns.pt'), map_location='cpu')
    test_data_path=test_opts.test_data_path
    output_path=test_opts.output_path
    os.makedirs(output_path, exist_ok=True)

    # origin data
        
    cates = os.listdir(test_data_path)
    for cate in cates:
        from_ims = os.listdir(os.path.join(test_data_path, cate))
        for j in tqdm(range(test_opts.n_images)):
            cr_dic=cr_dictionary[:,:test_opts.t]
            latents_buffer=[]
            for i in range(3):
                from_im_name = from_ims[i]
                from_im = Image.open(os.path.join(test_data_path, cate, from_im_name))
                from_im = from_im.convert('RGB')
                from_im = transform(from_im)
                latents = net.get_latents(from_im.unsqueeze(0).to("cuda").float())
                latents_buffer.append(latents[0])
            latents_buffer=torch.stack(latents_buffer)
            av_latent = torch.mean(latents_buffer, dim=0)
            n_dist = get_local_distribution(av_latent, cr_directions, ns, class_embeddings, test_opts.n_similar_cates)
            codes=sampler(net.ax.A, av_latent.unsqueeze(0), n_dist, cr_dic, alpha=test_opts.alpha)
            with torch.no_grad():
                res0 = net.decode(codes, randomize_noise=False, resize=opts.resize_outputs)
            res0 = tensor2im(res0[0])
            im_save_path = os.path.join(output_path, from_im_name+'_av_'+str(j)+'.jpg')
            Image.fromarray(np.array(res0)).save(im_save_path)
            for i in range(latents_buffer.SHAPE[0]):
                codes=sampler(net.ax.A, latents_buffer[i].unsqueeze(0), n_dist, cr_dic, alpha=test_opts.alpha)
                with torch.no_grad():
                    res0 = net.decode(codes, randomize_noise=False, resize=opts.resize_outputs)
                res0 = tensor2im(res0[0])
                im_save_path = os.path.join(output_path, from_im_name+'_'+str(i)+'_'+str(j)+'.jpg')
                Image.fromarray(np.array(res0)).save(im_save_path)





