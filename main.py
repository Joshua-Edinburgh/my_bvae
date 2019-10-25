# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:47:15 2019

@author: Joshua
"""

"""main.py"""

import argparse

import numpy as np
import torch

from solver import IVAE_Solver
from utils.basic import str2bool
from metrics import Metric_topsim

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = IVAE_Solver(args)
    net.iterated_learning()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter_per_gen', default=5000, type=int, help='maximum training iteration per generation')
    parser.add_argument('--max_gen', default=10, type=int, help='number of generations')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--z_dim', default=5, type=int, help='dimension of the representation z')
    parser.add_argument('--a_dim', default=40, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--nb_preENDE', default=100, type=int, help='Number of batches for pre-train encoder and decoder')
    parser.add_argument('--niter_preEN', default=2000, type=int, help='Number of max iterations for pre-train encoder')
    parser.add_argument('--niter_preDE', default=800, type=int, help='Number of max iterations for pre-train decoder')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')
    
    parser.add_argument('--save_step', default=1e4, type=int, help='number of iterations after which a checkpoint is saved')
    parser.add_argument('--metric_step',default=1e3, type=int, help='number of iterations after which R and top_sim metric saved')
    parser.add_argument('--top_sim_batches',default=1000,type=int, help='number of batches of sampling z when calculating top_sim and R')
    parser.add_argument('--save_gifs',default=True, type=str2bool, help='whether save the gifs')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--exp_name', default='learning_curves', type=str, help='name of the experiment')
    
    args = parser.parse_args()

    main(args)
'''
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = IVAE_Solver(args)
    loss_list = net.interact_train()
'''    
    #out_z, out_yc, out_x = net.gen_z(gen_size=args.nb_preENDE)
    
    #loss_table_EN = net.pre_train_EN(out_z, out_x)
    #loss_table_DE = net.pre_train_DE(out_z, out_x)


#    net = IVAE_Solver(args)
#    net.net_mode(False)
#    out_z, out_yc, out_x = net.gen_z(gen_size=args.nb_preENDE)
#    net.net_mode(True)
#    loss_table_EN = net.pre_train_EN(out_z, out_x)
#    loss_table_DE = net.pre_train_DE(out_z, out_x)
    
    
    #net.train()
    #net.load_checkpoint('last')
    #out_z,out_y,out_x = net.gen_z(10)




