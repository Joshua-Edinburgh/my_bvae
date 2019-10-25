#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:32:56 2019

@author: joshua
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from solver import IVAE_Solver, reconstruction_loss
from utils.basic import str2bool
from torch.autograd import Variable
from utils.basic import cuda
from model import reparametrize
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import torch.utils.data as Data # Dataset, DataLoader


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def show_images_grid(imgs_, num_images=25):
  ncols = int(np.ceil(num_images**0.5))
  nrows = int(np.ceil(num_images / ncols))
  _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
  axes = axes.flatten()

  for ax_i, ax in enumerate(axes):
    if ax_i < num_images:
      ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
    else:
      ax.axis('off')

def show_density(imgs):
  _, ax = plt.subplots()
  ax.imshow(imgs.mean(axis=0), interpolation='nearest', cmap='Greys_r')
  ax.grid('off')
  ax.set_xticks([])
  ax.set_yticks([])

def smooth_x(x,ratio=10):
    if type(x) != np.ndarray:
        x = np.asarray(x)
    new_x = np.zeros(x.shape)
    tmp = x[0]
    for i in range(x.size):
        tmp = (1-1/ratio)*tmp + 1/ratio*x[i]
        new_x[i] = tmp
    return new_x


def prepare_dataset(args, dataset_zip,smp_size=5000):
    def latent_to_index(latents):
      return np.dot(latents, latents_bases).astype(int)
    
    
    def sample_latent(size=1):
      samples = np.zeros((size, latents_sizes.size))
      for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)
    
      return samples
    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    
    
    latents_classes = dataset_zip['latents_classes']
    metadata = dataset_zip['metadata'][()]
    latents_sizes = metadata[b'latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1,])))        

    latents_sampled = sample_latent(size=smp_size)  
    #latents_sampled[:, -5] = 2       # Fix shape
    #latents_sampled[:, -4] = 3       # Fix size
    #latents_sampled[:, -3] = 20      # Fix rotation
    #latents_sampled[:, -2] = 16      # Fix pos_x
    #latents_sampled[:, -1] = 16      # Fix pos_y
    
    indices_sampled = latent_to_index(latents_sampled)
    out_x = imgs[indices_sampled]           # Size is smp_size*64*64        
    out_y = latents_values[indices_sampled,:] # Size is smp_size*64*64    
    out_yc = latents_sampled[:,:]

    
    perm_table = np.arange(0,out_y.shape[0],1)
    np.random.shuffle(perm_table)
    perm_y, perm_yc = out_y[perm_table], out_yc[perm_table]
    
    imgs = torch.from_numpy(out_x).unsqueeze(1)
    vals = torch.from_numpy(out_y).unsqueeze(1)
    clas = torch.from_numpy(out_yc).unsqueeze(1).int()
    data_set = Data.TensorDataset(imgs,vals,clas)

    orig_loader = Data.DataLoader(
                              dataset = data_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)

    perm_vals = torch.from_numpy(perm_y).unsqueeze(1)
    perm_clas = torch.from_numpy(perm_yc).unsqueeze(1).int() 
    perm_data_set = Data.TensorDataset(imgs, perm_vals, perm_clas)

    perm_loader = Data.DataLoader(
                              dataset = perm_data_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)

    return orig_loader, perm_loader    


def decoder_curves(args, data_loader, flag=False):    
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = IVAE_Solver(args)
    net.net_mode(True)
    out = False
    
    pre_DE_cnt = 0
    pbar = tqdm(total=net.niter_preDE)
    pbar.update(pre_DE_cnt)
    loss_table = []
    
    while not out:
        for x, y, yc in data_loader:
            x = Variable(cuda(x.float(), net.use_cuda))
            y = Variable(cuda(y.float(), net.use_cuda))
            yc = Variable(cuda(yc.long(), net.use_cuda))
            cut_yc = yc.squeeze(1)[:,1:]
            zero_matrix = Variable(cuda(torch.zeros(args.batch_size,args.z_dim,args.a_dim), net.use_cuda))
            z_onehot = zero_matrix.scatter_(2,cut_yc.unsqueeze(-1),1)
            
            x_recon = net.net._decode(z_onehot) 
            loss = reconstruction_loss(x, x_recon, 'bernoulli')
            loss_table.append(loss.data.item())
            
            net.optim_DE.zero_grad()
            loss.backward()
            net.optim_DE.step()
            
            if pre_DE_cnt >= net.niter_preDE:
                out = True
                break
            pre_DE_cnt += 1
            pbar.update(1)
            
    pbar.write("[Pretrain Encoder Finished]") 
    pbar.close() 
    sys.stdout.flush() 
    return loss_table

def encoder_curves(args,data_loader,flag=False):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = IVAE_Solver(args)
    net.net_mode(True)
    out = False
    
    pre_EN_cnt = 0
    pbar = tqdm(total=net.niter_preEN)
    pbar.update(pre_EN_cnt)
    loss_fun = torch.nn.CrossEntropyLoss()
    loss_table = []
    while not out:
        for x, y, yc in data_loader:
            x = Variable(cuda(x.float(), net.use_cuda))
            y = Variable(cuda(y.float(), net.use_cuda))
            yc = Variable(cuda(yc.long(), net.use_cuda))
            sub_hat = net.net._encode(x)
            sub1_hat,sub2_hat,sub3_hat, sub4_hat, sub5_hat = sub_hat[:,0,:],sub_hat[:,1,:],sub_hat[:,2,:],sub_hat[:,3,:],sub_hat[:,4,:]
            loss = loss_fun(sub1_hat,yc[:,0,1]) + loss_fun(sub2_hat,yc[:,0,2]) + loss_fun(sub3_hat,yc[:,0,3]) +\
                    loss_fun(sub4_hat,yc[:,0,4]) + loss_fun(sub5_hat,yc[:,0,5])
                 
            loss_table.append(loss.data.item())
            
            net.optim_EN.zero_grad()
            loss.backward()
            net.optim_EN.step()
              
            if pre_EN_cnt >= net.niter_preEN:
                out = True
                break
            pre_EN_cnt += 1
            pbar.update(1)
    pbar.write("[Pretrain Encoder Finished]") 
    pbar.close() 
    sys.stdout.flush()  

    return loss_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter_per_gen', default=100, type=int, help='maximum training iteration per generation')
    parser.add_argument('--max_gen', default=10, type=int, help='number of generations')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')

    parser.add_argument('--z_dim', default=5, type=int, help='dimension of the representation z')
    parser.add_argument('--a_dim', default=40, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--nb_preENDE', default=100, type=int, help='Number of batches for pre-train encoder and decoder')
    parser.add_argument('--niter_preEN', default=5000, type=int, help='Number of max iterations for pre-train encoder')
    parser.add_argument('--niter_preDE', default=5000, type=int, help='Number of max iterations for pre-train decoder')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')
    
    parser.add_argument('--save_step', default=1e4, type=int, help='number of iterations after which a checkpoint is saved')
    parser.add_argument('--metric_step',default=1e4, type=int, help='number of iterations after which R and top_sim metric saved')
    parser.add_argument('--top_sim_batches',default=1000,type=int, help='number of batches of sampling z when calculating top_sim and R')
    parser.add_argument('--save_gifs',default=True, type=str2bool, help='whether save the gifs')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--exp_name', default='learning_curves', type=str, help='name of the experiment')
    
    args = parser.parse_args()

    data_not_load = True
    if data_not_load:
        root = os.path.join(args.dset_dir,'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        dataset_zip = np.load(root, allow_pickle=True, encoding='bytes')
        data_not_load = False
        
    data_loader,perm_loader = prepare_dataset(args, dataset_zip,smp_size=5000)
      
    loss_table1 = decoder_curves(args,data_loader,False)
    loss_table2 = decoder_curves(args,perm_loader,False)
#    loss_table1 = encoder_curves(args,data_loader,False)
#    loss_table2 = encoder_curves(args,perm_loader,False)
    
    x_axis = np.arange(0,len(loss_table1),1)
    plt.plot(x_axis,smooth_x(loss_table1,20),'r',label='origin')
    plt.plot(x_axis,smooth_x(loss_table2,20),'b',label='permut')
    plt.legend()
    plt.grid(True)
    plt.show()
    