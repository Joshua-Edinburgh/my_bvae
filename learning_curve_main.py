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
from utils.dataset import z_values_dsprites, z_values_3dshapes
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

def clas_to_vals(args, clas):
    if args.data_type.lower() == 'dsprites':
        z_values = z_values_dsprites        
    elif args.data_type.lower() == '3dshapes':
        z_values = z_values_3dshapes 
        
    data_len, dim_len = clas.shape[0], clas.shape[1]
    vals = np.zeros(clas.shape)
    key_list = list(z_values.keys())
    
    for dim_idx in range(dim_len):
        tmp_dim_vals = z_values[key_list[dim_idx]]
        for idx in range(data_len):
            tmp_clas = clas[idx, dim_idx]
            vals[idx, dim_idx] = tmp_dim_vals[tmp_clas]        
    
    return vals


def data_to_batches(args, zyc_pairs, shuffle=True):
    z, yc = zyc_pairs['z'], zyc_pairs['yc']  
    yv = clas_to_vals(args, yc)
    batch_size = args.batch_size
    data_length = z.shape[0]
    permut_mask = np.arange(0,data_length,1)
    np.random.shuffle(permut_mask)
    z, yc, yv = z[permut_mask], yc[permut_mask], yv[permut_mask]
    z = torch.from_numpy(z)
    yc = torch.from_numpy(yc)
    yv = torch.from_numpy(yv)

    
    if args.model_type.lower() in ['fcvae','cvae']:
        z = Variable(cuda(z.long(), args.cuda))
    elif args.model_type.lower() in ['fvae','bvae']:
        z = Variable(cuda(z.float(), args.cuda)).unsqueeze(-1).unsqueeze(-1)
    yc = Variable(cuda(yc.long(), args.cuda))
    yv = Variable(cuda(yv.float(), args.cuda))
    
    z_batch, yc_batch, yv_batch = [], [], []
    out = False
    idx = 0
    while not out:
        if idx+batch_size>data_length:
            out = True
            break
        z_batch.append(z[idx:idx+batch_size,:])
        yc_batch.append(yc[idx:idx+batch_size,:])
        yv_batch.append(yv[idx:idx+batch_size,:])
        idx = idx+batch_size
    
    return z_batch, yc_batch, yv_batch

def encoder_curves(args, out_z, out_yc, out_yv, flag=False):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    solver = IVAE_Solver(args)
    solver.net_mode(True)
    if args.model_type.lower() in ['fcvae','cvae']:
        loss_table = solver.pre_train_EN(out_z, out_yc)
    elif args.model_type.lower() in ['fvae','bvae']:
        loss_table = solver.pre_train_EN(out_z, out_yc)
    return loss_table


def decoder_curves(args, out_z, out_yc, out_yv, flag=False):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    solver = IVAE_Solver(args)
    solver.net_mode(True)
    if args.model_type.lower() in ['fcvae','cvae']:
        loss_table = solver.pre_train_DE(out_z, out_yc)
    elif args.model_type.lower() in ['fvae','bvae']:
        loss_table = solver.pre_train_DE(out_z, out_yc)
    return loss_table

def gen_perfect_z(args, zyc_pairs):
    zyc_pairs_perfect = {}
    z_dim = args.z_dim
    yc = zyc_pairs['yc'] 
    yv = clas_to_vals(args, yc)
    data_len = yc.shape[0]
    dim_len = yc.shape[1]
    zeros = np.zeros((data_len, z_dim-dim_len))
    if args.model_type.lower() in ['fcvae','cvae']:
        tmp_z = np.concatenate((yc,zeros),axis=1)
    elif args.model_type.lower() in ['fvae','bvae']:
        tmp_z = np.concatenate((yv,zeros),axis=1)
    zyc_pairs_perfect['z'] = tmp_z
    zyc_pairs_perfect['yc'] = yc
    return zyc_pairs_perfect

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')
    
    parser.add_argument('--model_type', default='FVAE', type=str, help='BVAE, CVAE, FVAE, FCVAE or VQVAE')
    parser.add_argument('--data_type', default='3dshapes', type=str, help='dsprites, 3dshapes or colormnist')

    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=12345, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter_per_gen', default=10, type=int, help='maximum training iteration per generation')
    parser.add_argument('--max_gen', default=2, type=int, help='number of generations')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--a_dim', default=40, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--gamma', default=30, type=float, help='gamma parameter for Factor-VAE')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--nb_preENDE', default=10, type=int, help='Number of batches for pre-train encoder and decoder')
    parser.add_argument('--niter_preEN', default=1500, type=int, help='Number of max iterations for pre-train encoder')
    parser.add_argument('--niter_preDE', default=1000, type=int, help='Number of max iterations for pre-train decoder')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')
    
    parser.add_argument('--save_step', default=1e5, type=int, help='number of iterations after which a checkpoint is saved')
    parser.add_argument('--metric_step',default=1e4, type=int, help='number of iterations after which R and top_sim metric saved')
    parser.add_argument('--top_sim_batches',default=1000,type=int, help='number of batches of sampling z when calculating top_sim and R')
    parser.add_argument('--save_gifs',default=False, type=str2bool, help='whether save the gifs')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--exp_name', default='test_learning_curves', type=str, help='name of the experiment')
    
    args = parser.parse_args()

    zyc_pairs1 = np.load('test_learning_curves/zyc_pairs_gen1_it1350001.npz')
    zyc_pairs2 = np.load('test_learning_curves/zyc_pairs_gen1_it100001.npz')
    zyc_pairs_perfect = gen_perfect_z(args, zyc_pairs1)
   
    out_z1, out_yc1, out_yv1 = data_to_batches(args, zyc_pairs1)
    out_z2, out_yc2, out_yv2 = data_to_batches(args, zyc_pairs2)
    out_zp, out_ycp, out_yvp = data_to_batches(args, zyc_pairs_perfect)
    loss_table1 = encoder_curves(args, out_z1, out_yc1, out_yv1)
    loss_table2 = encoder_curves(args, out_z2, out_yc2, out_yv2)
#    loss_tablep = encoder_curves(args, out_zp, out_ycp, out_yvp)
 
#    loss_table1 = decoder_curves(args, out_z1, out_yc1, out_yv1)
#    loss_table2 = decoder_curves(args, out_z2, out_yc2, out_yv2)
#    loss_tablep = decoder_curves(args, out_zp, out_ycp, out_yvp)
    
    x = np.arange(0,len(loss_table1),1)
    plt.plot(x, loss_table1,label='High-DCI')
    plt.plot(x, loss_table2,label='Low-DCI')
#    plt.plot(x,loss_tablep,label='Perfect-DCI')
    plt.ylim(0,2)
    plt.legend()
  
        
