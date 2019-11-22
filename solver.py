#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:42:56 2019

@author: xiayezi
"""
import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm

import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from utils.basic import cuda, grid2gif
from model import BetaVAE_H, reparametrize
from utils.dataset import return_data
from metrics import Metric_R, Metric_topsim, unpack_batch_x, unpack_batch_zoy

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()
        

class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter_per_gen = args.max_iter_per_gen
        self.max_gen = args.max_gen
        self.global_iter = 0
        self.global_gen = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        
        self.nb_preENDE = args.nb_preENDE
        self.niter_preEN = args.niter_preEN
        self.niter_preDE = args.niter_preDE

        self.nc = 1
        self.decoder_dist = 'bernoulli'
        
        net = BetaVAE_H

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))
        self.optim_EN = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))
        self.optim_DE = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))


        self.exp_name = args.exp_name
        self.ckpt_dir = os.path.join('exp_results/'+args.exp_name,args.ckpt_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)
            
        self.metric_dir = os.path.join('exp_results/'+args.exp_name+'/metrics')
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)

        self.imgs_dir = os.path.join('exp_results/'+args.exp_name+'/images')
        if not os.path.exists(self.imgs_dir):
            os.makedirs(self.imgs_dir)
            
        self.save_step = args.save_step
        self.metric_step = args.metric_step
        self.metric_topsim = Metric_topsim(args)
        self.top_sim_batches = args.top_sim_batches
        self.metric_R = Metric_R(args)
        self.save_gifs = args.save_gifs

        self.dset_dir = args.dset_dir
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        self.gather = DataGather()      

    def iterated_learning(self):
        out_z = []
        out_x = []
        for gen_idx in range(int(self.max_gen)):
            self.global_gen += 1
            print('\n======= This is generation{:>2d}/{:>2d}  ======'.format(gen_idx+1,self.max_gen))
            if gen_idx != 0:
                print('------ Pretraining Encoder {:>2d}/{:>2d} ------'.format(gen_idx+1,self.max_gen))
                sys.stdout.flush()
                self.net.encoder_init()
                self.pre_train_EN(out_z, out_x)
                         
            if gen_idx != 0:
                print('------ Pretraining Decoder {:>2d}/{:>2d} ------'.format(gen_idx+1,self.max_gen))
                sys.stdout.flush() 
                self.net.decoder_init()
                self.pre_train_DE(out_z, out_x)
               
            print('------ Interactive Training {:>2d}/{:>2d} -----'.format(gen_idx+1,self.max_gen))
            sys.stdout.flush()
            self.interact_train()
                       
            print('------- Data Generating {:>2d}/{:>2d} ---------'.format(gen_idx+1,self.max_gen))
            sys.stdout.flush()  
            out_z, _, out_x = self.gen_z(self.nb_preENDE)
            
    def pre_train_EN(self, out_z, out_x):
        self.net_mode(True)
        out = False
        pre_EN_cnt = 0
        pbar = tqdm(total=self.niter_preEN)
        pbar.update(pre_EN_cnt)
        loss_fun = torch.nn.MSELoss()
        loss_table = []
        while not out:  
            for z,x in zip(out_z,out_x):  
                x = Variable(cuda(x.float(), self.use_cuda))
                z = Variable(cuda(z.float(), self.use_cuda))
                distributions = self.net._encode(x)
                mu = distributions[:, :self.z_dim]
                logvar = distributions[:, self.z_dim:]
                z_hat = reparametrize(mu, logvar)    
                loss = loss_fun(z_hat,z)
                loss_table.append(loss.data.item())
                
                self.optim_EN.zero_grad()
                loss.backward()
                self.optim_EN.step()
                
                if pre_EN_cnt >= self.niter_preEN:
                    out = True
                    break                
                pre_EN_cnt += 1
                pbar.update(1)
        pbar.write("[Pretrain Encoder Finished]")
        pbar.close() 
        sys.stdout.flush()               
        return loss_table
    
    def pre_train_DE(self,out_z,out_x):
        self.net_mode(True)        
        out = False
        pre_DE_cnt = 0
        pbar = tqdm(total=self.niter_preDE)
        pbar.update(pre_DE_cnt)
        loss_table = []
        
        while not out:
            for z, x in zip(out_z, out_x):
                x = Variable(cuda(x.float(), self.use_cuda))
                z = Variable(cuda(z.float(), self.use_cuda))
                x_recon = self.net._decode(z)     
                loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                loss_table.append(loss.data.item())
                
                self.optim_DE.zero_grad()
                loss.backward()
                self.optim_DE.step()
                
                if pre_DE_cnt >= self.niter_preDE:
                    out = True
                    break
                pre_DE_cnt += 1
                pbar.update(1)                
        pbar.write("[Pretrain Decoder Finished]")
        pbar.close()   
        sys.stdout.flush()               
        return loss_table                
    
    def interact_train(self):
        self.net_mode(train=True)
        out = False
        
        indx_list = []
        loss_list = []
        corr_list = []
        dist_list = []
        comp_list = []
        info_list = []
        R_list = []
        recon_loss_list = []
        bvae_loss_list = []

        pbar = tqdm(total=self.max_iter_per_gen)
        pbar.update(0)
        local_iter = 0
        with open(self.metric_dir+'/results.txt','a') as f:
            f.write('====== Experiment name: '+self.exp_name+'==============\n')
            
        while not out:
            for x,y in self.data_loader:
                self.global_iter += 1
                local_iter += 1
                pbar.update(1)

                x = Variable(cuda(x.float(), self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                beta_vae_loss = recon_loss + self.beta*total_kld
                
                recon_loss_list.append(recon_loss.data.item())
                bvae_loss_list.append(beta_vae_loss.data.item())
                
                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                if False:#self.global_iter%(self.metric_step*5) == 0:
                    out_fz, out_y, out_x = self.gen_z(self.top_sim_batches*10, fullz=True)
                    fz_upk = torch.stack(out_fz).view(-1,self.z_dim*2).cpu()
                    y_upk = torch.stack(out_y).view(-1,5).cpu()
                    x_upk = torch.stack(out_x).view(-1,64,64).cpu()
                    np.savez(self.metric_dir+'/saved_xyz_'+str(self.global_iter)+'.npz',
                             out_fz = np.asarray(fz_upk),     # Here out_z is the distribution ([mu:sigma])
                             out_y = np.asarray(y_upk),
                             out_x = np.asarray(x_upk)
                             )                              
                    #test_z = np.load(os.path.join(net.metric_dir+'/saved_xyz0.npz'))
                    
                if self.global_iter%self.metric_step == 0:
                    out_z, out_y, _ = self.gen_z(self.top_sim_batches)                    
                    corr = self.metric_topsim.top_sim_zy(out_z[:20],out_y[:20])
                    dist, comp, info, R = self.metric_R.dise_comp_info(out_z,out_y,'random_forest')
                    indx_list.append(self.global_iter)
                    loss_list.append(recon_loss.data.item())
                    corr_list.append(corr)
                    dist_list.append(dist)
                    comp_list.append(comp)
                    info_list.append(info)
                    R_list.append(R)
                    
                    
                    
                    #print('======================================')
                    with open(self.metric_dir+'/results.txt','a') as f:
                        f.write('\n [{:0>7d}] \t loss:{:.3f} \t corr:{:.3f} \t dise:{:.3f} \t comp:{:.3f}\t info:{:.3f}'.format(
                                self.global_iter, recon_loss.data.item(), corr, dist[-1], comp[-1],info[-1]))
                    #print('======================================')
                if self.global_iter%self.save_step == 1:
                    self.save_checkpoint('last')
                    #pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                    if self.save_gifs:
                        self.save_gif()

                if self.global_iter%50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if local_iter >= self.max_iter_per_gen:
                    out = True
                    break
                
        np.savez(self.metric_dir+'/metrics_gen'+str(self.global_gen)+'.npz',
                 indx = np.asarray(indx_list),       # (len,)
                 loss = np.asarray(loss_list),       # (len,)
                 corr = np.asarray(corr_list),       # (len,)
                 dist = np.asarray(dist_list),       # (len, z_dim+1)
                 comp = np.asarray(comp_list),       # (len, y_dim+1)
                 info = np.asarray(info_list),       # (len, y_dim+1)
                 R = np.asarray(R_list))             # (len, z_dim, y_dim)
        pbar.write("[Training Finished]")
        pbar.close()
        sys.stdout.flush()  
        return recon_loss_list, bvae_loss_list
        
    def gen_z(self, gen_size=10, fullz=False):
        '''
            Randomly sample x from dataloader, feed it to encoder, generate z
            Return z and true latent value
            @ out_z should be a list with length equals gen_size, each object has 
            size B*z_dim
            @ out_y should be a list with length equals gen_size, each object has
            size B*6
        '''
        self.net_mode(train=False)
        out = False
        gen_cnt = 0
        #pbar = tqdm(total=gen_size)
        #pbar.update(gen_cnt)
        out_z = []
        out_y = []
        out_x = []
        out_distr_list = []
        
        while not out:
            for x,y in self.data_loader:
                out_y.append(y.squeeze(1)[:,1:])
                out_x.append(x)
                #pbar.update(1)
                gen_cnt += 1
                x = Variable(cuda(x.float(), self.use_cuda))
                out_distri = self.net.encoder(x).data
                mu = out_distri[:, :self.z_dim]
                logvar = out_distri[:, self.z_dim:]    
                out_z.append(reparametrize(mu, logvar))  
                out_distr_list.append(out_distri)
                if gen_cnt >= gen_size:
                    out = True
                    break
        self.net_mode(train=True)
        if fullz == True:
            return out_distr_list, out_y, out_x
        else:
            return out_z, out_y, out_x

    def save_gif(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random
        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)
        n_dsets = len(self.data_loader.dataset)        
        rand_idx = random.randint(1, n_dsets-1)
        
        random_img, _ = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img.float(), self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]  
        
        fixed_idx1 = 87040 # square
        fixed_idx2 = 332800 # ellipse
        fixed_idx3 = 578560 # heart

        fixed_img1, _ = self.data_loader.dataset.__getitem__(fixed_idx1)
        fixed_img1 = Variable(cuda(fixed_img1.float(), self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

        fixed_img2, _ = self.data_loader.dataset.__getitem__(fixed_idx2)
        fixed_img2 = Variable(cuda(fixed_img2.float(), self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

        fixed_img3, _ = self.data_loader.dataset.__getitem__(fixed_idx3)
        fixed_img3 = Variable(cuda(fixed_img3.float(), self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

        Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
             'fixed_heart':fixed_img_z3, 'random_img':random_img_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()

        output_dir = os.path.join(self.imgs_dir, str(self.global_iter))
        os.makedirs(output_dir, exist_ok=True)
        gifs = torch.cat(gifs)
        gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
        for i, key in enumerate(Z.keys()):
            for j, val in enumerate(interpolation):
                save_image(tensor=gifs[i][j].cpu(),
                           filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                           nrow=self.z_dim, pad_value=1)

            grid2gif(os.path.join(output_dir, key+'*.jpg'),
                     os.path.join(output_dir, key+'.gif'), delay=10)    
        self.net_mode(train=True)
        
    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()
        
    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))        
        
        
        
        
        
        
        
        



