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
from model import *
from utils.dataset import return_data_dsprites, return_data_3dshapes
from metrics import unpack_batch_y, unpack_batch_z, Metric_DCI, Metric_topsim, Metric_Factor
from torch.distributions.one_hot_categorical import OneHotCategorical


def z_to_onehot(z,z_dim,a_dim,use_cuda):
    z = z.long()
    b_size = z.size(0)
    for i in range(b_size):
        tmp_onehot = Variable(cuda(torch.FloatTensor(z_dim,a_dim),use_cuda))
        tmp_onehot.zero_()
        tmp_onehot = tmp_onehot.scatter_(1,z[i,:].unsqueeze(1),1)
        tmp_onehot = tmp_onehot.unsqueeze(0)
        if i == 0:
            z_onehot = tmp_onehot
        else:
            z_onehot = torch.cat((z_onehot,tmp_onehot),dim=0)   
    return z_onehot.view(-1,z_dim*a_dim)



def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)   

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

def KLD_Gaussian(mu, logvar):
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

def KLD_Catagorical(sftmx):
    '''
        Assume p(z) has uniform catagorical distribution
    '''
    batch_size, z_dim, a_dim = sftmx.shape
    sftmx = sftmx.view(batch_size,-1)
    kl1 = (sftmx*torch.log(sftmx+1e-20))
    kl2 = (sftmx*np.log(a_dim+1e-20))              
    total_kld = torch.sum(kl1+kl2,dim=1).mean()

    return total_kld
        

class IVAE_Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter_per_gen = args.max_iter_per_gen
        self.max_gen = args.max_gen
        self.global_iter = 0
        self.global_gen = 0
        self.data_type = args.data_type
        self.model_type = args.model_type

        self.z_dim = args.z_dim
        self.a_dim = args.a_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.gumbel_tmp = 1.0
        
        self.nb_preENDE = args.nb_preENDE
        self.niter_preEN = args.niter_preEN
        self.niter_preDE = args.niter_preDE
        
        if (self.data_type).lower()=='dsprites':
            self.data_loader = return_data_dsprites(args)
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif (self.data_type).lower()=='3dshapes':
            self.data_loader = return_data_3dshapes(args)
            self.nc = 3
            self.decoder_dist = 'gaussian'
        
        if (self.model_type).lower()=='bvae':
            net = BetaVAE_H
            self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        elif (self.model_type).lower()=='cvae':
            net = CVAE
            self.net = cuda(net(self.z_dim, self.nc, self.a_dim), self.use_cuda)
        elif (self.model_type).lower() in ['fvae','fcvae']:
            self.D = cuda(F_Discriminator(self.z_dim), self.use_cuda)
            self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr,betas=(self.beta1, self.beta2))
            if (self.model_type).lower() == 'fvae':
                net = FVAE
                self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
            else:
                net = FCVAE
                self.net = cuda(net(self.z_dim, self.nc,self.a_dim), self.use_cuda)
        elif (self.model_type).lower()=='vqvae':
            net = VQVAE
            self.net = cuda(net(self.z_dim, self.nc, self.a_dim, 1), self.use_cuda)
        else:
            raise('model_type should be BVAE, CVAE, FVAE, FCVAE or VQVAE')

        self.MSE_Loss = nn.MSELoss()    
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,betas=(self.beta1, self.beta2))
        self.optim_EN = optim.Adam(self.net.parameters(), lr=self.lr,betas=(self.beta1, self.beta2))
        self.optim_DE = optim.Adam(self.net.parameters(), lr=self.lr,betas=(self.beta1, self.beta2))

#        self.optim = optim.RMSprop(self.net.parameters(), lr=self.lr)
#        self.optim_EN = optim.RMSprop(self.net.parameters(), lr=self.lr)
#        self.optim_DE = optim.RMSprop(self.net.parameters(), lr=self.lr)

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
        #self.metric_topsim = Metric_topsim(args)
        self.top_sim_batches = args.top_sim_batches
        self.metric_DCI = Metric_DCI(args)
        self.metric_Factor = Metric_Factor(args)
        self.save_gifs = args.save_gifs

        self.dset_dir = args.dset_dir
        self.batch_size = args.batch_size
    
            
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
                loss_table1 = self.pre_train_EN(out_z, out_x)
                         
            if gen_idx != 0:
                print('------ Pretraining Decoder {:>2d}/{:>2d} ------'.format(gen_idx+1,self.max_gen))
                sys.stdout.flush() 
                self.net.decoder_init()
                loss_table2 = self.pre_train_DE(out_z, out_x)
               
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
        loss_fun = torch.nn.CrossEntropyLoss()
        loss_table = []
        while not out:  
            for z,x in zip(out_z,out_x): 
                loss = Variable(cuda(x.float(), self.use_cuda))
                x = Variable(cuda(x.float(), self.use_cuda))
                z_hat = self.net._encode(x).view(-1,self.z_dim,self.a_dim).transpose(1,2)
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
                
                z_onehot = z_to_onehot(z,self.z_dim,self.a_dim,self.use_cuda)
                
                x_recon = self.net._decode(z_onehot)     
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
        Fscore_list = []
        DCI_list = []
        R_list = []

        pbar = tqdm(total=self.max_iter_per_gen)
        pbar.update(0)
        local_iter = 0
        with open(self.metric_dir+'/results.txt','a') as f:
            f.write('\n====== Experiment name: '+self.exp_name+'==============')
            
        while not out:
            for x,y,yc in self.data_loader:
                self.global_iter += 1
                local_iter += 1
                pbar.update(1)

                if (self.data_type).lower()=='dsprites':
                    x = Variable(cuda(x.float(), self.use_cuda))
                elif (self.data_type).lower()=='3dshapes':
                    x = Variable(cuda(x.float()/255, self.use_cuda))
                
                if (self.model_type).lower()=='bvae':
                    x_recon, mu, logvar = self.net(x)
                    recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                    total_kld, _dim_wise_kld, _mean_kld = KLD_Gaussian(mu, logvar)
                    beta_vae_loss = recon_loss + self.beta*total_kld
                    loss_list.append(recon_loss.data.item())
                    self.optim.zero_grad()
                    beta_vae_loss.backward()
                    self.optim.step()     
                    
                elif (self.model_type).lower()=='cvae':
                    x_recon, sftmx = self.net(x)
                    recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                    total_kld = KLD_Catagorical(sftmx)
                    cvae_loss = recon_loss + self.beta*total_kld
                    loss_list.append(recon_loss.data.item())    
                    self.optim.zero_grad()
                    cvae_loss.backward()
                    self.optim.step()

                elif (self.model_type).lower() in ['fvae','fcvae']:
                    half = int(self.batch_size/2)
                    ones = Variable(cuda(torch.ones(half, dtype=torch.long),self.use_cuda))
                    zeros = Variable(cuda(torch.zeros(half, dtype=torch.long),self.use_cuda))                    
                    x1, x2 = x[:half], x[half:]
                    y1, y2 = y[:half], y[half:]
                    yc1, yc2 = yc[:half], yc[half:]     
                    if (self.model_type).lower() == 'fvae':
                        x_recon, mu, logvar, z = self.net(x1)
                        total_kld, _, _ = KLD_Gaussian(mu, logvar) 
                    elif (self.model_type).lower() == 'fcvae':
                        x_recon, sftmx, z = self.net(x1)
                        total_kld = KLD_Catagorical(sftmx)  
                    recon_loss = reconstruction_loss(x1, x_recon, self.decoder_dist)                  
                    D_z = self.D(z)
                    tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
                    fvae_loss = recon_loss + total_kld + self.gamma*tc_loss
                    loss_list.append(recon_loss.data.item()) 
                    self.optim.zero_grad()
                    fvae_loss.backward(retain_graph=True)
                    self.optim.step()                                    
                    z_prime = self.net(x2, no_dec=True)
                    z_pperm = permute_dims(z_prime).detach()
                    D_z_pperm = self.D(z_pperm)
                    D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))
                    self.optim_D.zero_grad()
                    D_tc_loss.backward()
                    self.optim_D.step()
                    
                elif (self.model_type).lower() in ['vqvae']:
                    x_recon, z_enc, z_dec, z_enc_for_embd = self.net(x)
                    recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                    loss_list.append(recon_loss.data.item())    
                    z_sgembd_loss = self.MSE_Loss(z_enc, z_dec.detach())
                    sgz_embd_loss = self.MSE_Loss(self.net._modules['embd'].weight, z_enc_for_embd.detach())
                    vqvae_loss = recon_loss + sgz_embd_loss + self.beta*z_sgembd_loss
                    self.optim.zero_grad()
                    vqvae_loss.backward(retain_graph=True)
                    z_enc.backward(self.net.grad_for_encoder)
                    self.optim.step()
                    
                    

                if self.global_iter%2000 == 2:
                    self.gumbel_tmp = np.max((0.5,np.exp((-5e-5)*self.global_iter)))
                    self.net.gumbel_tmp = self.gumbel_tmp
                    
                if self.global_iter%self.metric_step == 1:
                    out_z,out_yc = self.gen_z(self.top_sim_batches)
                    self.save_z_yc_pairs(out_z, out_yc)
                    corr = 0.
                    #corr = self.metric_topsim.top_sim_zy(out_z[:10],out_yc[:10])
                    Fscore = self.metric_Factor.get_score(out_z,out_yc)
                    dist, comp, info, DCI, R  = self.metric_DCI.dise_comp_info(out_z,out_yc,'random_forest')                    
                    indx_list.append(self.global_iter)
                    loss_list.append(recon_loss.data.item())
                    corr_list.append(corr)
                    dist_list.append(dist)
                    comp_list.append(comp)
                    info_list.append(info)
                    DCI_list.append(DCI)
                    Fscore_list.append(Fscore)
                    R_list.append(R)
                    #print('======================================')
                    with open(self.metric_dir+'/results.txt','a') as f:
                        f.write('\n [{:0>7d}] \t loss:{:.3f} \t dise:{:.3f} \t comp:{:.3f}\t info:{:.3f} \t DCI:{:.3f}\t FScore:{:.3f}'.format(
                                self.global_iter, recon_loss.data.item(), dist[-1], comp[-1],info[-1],DCI,Fscore))
                    #print('======================================')
                if self.global_iter%self.save_step == 2:
                    if self.save_gifs:
                        if (self.data_type).lower()=='dsprites':
                            self.save_gif_dsprites()
                        elif (self.data_type).lower()=='3dshapes':
                            self.save_gif_3dshapes()
                    
                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last')


                if self.global_iter%50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if local_iter >= self.max_iter_per_gen:
                    out = True
                    break
        # ============== End of one generation ================================
        out_z,out_yc = self.gen_z(self.top_sim_batches,all_data = True)
        self.save_z_yc_pairs(out_z, out_yc)        
        np.savez(self.metric_dir+'/metrics_gen'+str(self.global_gen)+'.npz',
                 indx = np.asarray(indx_list),       # (len,)
                 loss = np.asarray(loss_list),       # (len,)
                 corr = np.asarray(corr_list),       # (len,)
                 dist = np.asarray(dist_list),       # (len, z_dim+1)
                 comp = np.asarray(comp_list),       # (len, y_dim+1)
                 info = np.asarray(info_list),       # (len, y_dim+1)
                 Fscore = np.asarray(Fscore_list),   # (len,)
                 DCI = np.asarray(DCI_list),         # (len,)
                 R = np.asarray(R_list))             # (len, z_dim, y_dim)
        pbar.write("[Training Finished]")
        pbar.close()
        sys.stdout.flush()  
        return loss_list
    def gen_z(self, gen_size=10, all_data = False):
        '''
            Randomly sample x from dataloader, feed it to encoder, generate z
            Return z and true latent value
            @ out_z should be a list with length equals gen_size, each object has 
            size B*z_dim*a_dim
            @ out_y should be a list with length equals gen_size, each object has
            size B*6
        '''
        self.net_mode(train=False)
        out = False
        gen_cnt = 0
        out_z = []          # One hot shape [B,z_dim,a_dim], sampled
        out_yc = []
        
        while not out:
            for x,y,yc in self.data_loader:
                out_yc.append(yc.squeeze(1)[:,1:])
                gen_cnt += 1
                if (self.data_type).lower()=='dsprites':
                    x = Variable(cuda(x.float(), self.use_cuda))
                elif (self.data_type).lower()=='3dshapes':
                    x = Variable(cuda(x.float()/255, self.use_cuda))
                z = self.net.fd_gen_z(x)  
                out_z.append(z)                         
                if (all_data==False) and (gen_cnt >= gen_size):
                    out = True
                    break
                elif all_data == True:
                    out = False
            out = True
        self.net_mode(train=True)
        return out_z, out_yc

    def save_gif_dsprites(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random
        if (self.model_type).lower() in ['bvae','fvae','vqvae']:
            interpolation = torch.arange(-limit, limit+0.1, inter)
        elif (self.model_type).lower() in ['cvae','fcvae']:
            interpolation = torch.range(0,self.a_dim-1,1)
        n_dsets = len(self.data_loader.dataset)        
        rand_idx = random.randint(1, n_dsets-1)
        
        random_img, _, _ = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img.float(), self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = self.net.fd_gen_z(random_img)
        
        fixed_idx1 = 87040 # square
        fixed_idx2 = 332800 # ellipse
        fixed_idx3 = 578560 # heart

        fixed_img1, _, _ = self.data_loader.dataset.__getitem__(fixed_idx1)
        fixed_img1 = Variable(cuda(fixed_img1.float(), self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z1 = self.net.fd_gen_z(fixed_img1)
        

        fixed_img2, _, _= self.data_loader.dataset.__getitem__(fixed_idx2)
        fixed_img2 = Variable(cuda(fixed_img2.float(), self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z2 = self.net.fd_gen_z(fixed_img2)

        fixed_img3, _, _ = self.data_loader.dataset.__getitem__(fixed_idx3)
        fixed_img3 = Variable(cuda(fixed_img3.float(), self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z3 = self.net.fd_gen_z(fixed_img3)

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
                    if (self.model_type).lower() in ['bvae','fvae']:
                        sample = F.sigmoid(self.net._decode(z)).data
                    elif (self.model_type).lower() in ['vqvae']:
                        z_reshape = self.net.find_nearest(z,self.net.embd.weight).view(-1,self.z_dim,1,1)
                        sample = F.sigmoid(self.net._decode(z_reshape)).data
                    elif (self.model_type).lower() in ['cvae','fcvae']:
                        z_onehot = z_to_onehot(z,self.z_dim,self.a_dim,self.use_cuda)
                        sample = F.sigmoid(self.net._decode(z_onehot)).data
                
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

    def save_gif_3dshapes(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random
        if (self.model_type).lower() in ['bvae','fvae','vqvae']:
            interpolation = torch.arange(-limit, limit+0.1, inter)
        elif (self.model_type).lower() in ['cvae','fcvae']:
            interpolation = torch.range(0,self.a_dim-1,1)
        n_dsets = len(self.data_loader.dataset)        
        rand_idx = random.randint(1, n_dsets-1)
        
        random_img, _, _ = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img.float()/255, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = self.net.fd_gen_z(random_img)
        
        fixed_idx1 = 5940  # shape1     [0,1,2,3,0,0]
        fixed_idx2 = 53955 # shape2     [1,1,2,3,1,0]
        fixed_idx3 = 169155 # shape3    [3,5,2,3,2,0]
        fixed_idx4 = 243405 # shape4      [5,0,7,0,3,0]

        fixed_img1, _, _ = self.data_loader.dataset.__getitem__(fixed_idx1)
        fixed_img1 = Variable(cuda(fixed_img1.float()/255, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z1 = self.net.fd_gen_z(fixed_img1)
        

        fixed_img2, _, _= self.data_loader.dataset.__getitem__(fixed_idx2)
        fixed_img2 = Variable(cuda(fixed_img2.float()/255, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z2 = self.net.fd_gen_z(fixed_img2)

        fixed_img3, _, _ = self.data_loader.dataset.__getitem__(fixed_idx3)
        fixed_img3 = Variable(cuda(fixed_img3.float()/255, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z3 = self.net.fd_gen_z(fixed_img3)

        fixed_img4, _, _ = self.data_loader.dataset.__getitem__(fixed_idx4)
        fixed_img4 = Variable(cuda(fixed_img4.float()/255, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z4 = self.net.fd_gen_z(fixed_img4)

        Z = {'fixed_s1':fixed_img_z1, 'fixed_s2':fixed_img_z2,
             'fixed_s3':fixed_img_z3, 'fixed_s4':fixed_img_z4}
             #'random_img':random_img_z}
        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()         
                for val in interpolation:
                    z[:, row] = val                     
                    if (self.model_type).lower() in ['bvae','fvae']:
                        sample = F.sigmoid(self.net._decode(z)).data.transpose(-1,-2)
                    elif (self.model_type).lower() in ['vqvae']:
                        z_reshape = self.net.find_nearest(z,self.net.embd.weight).view(-1,self.z_dim,1,1)
                        sample = F.sigmoid(self.net._decode(z_reshape)).data.transpose(-1,-2)
                    elif (self.model_type).lower() in ['cvae','fcvae']:
                        z_onehot = z_to_onehot(z,self.z_dim,self.a_dim,self.use_cuda)
                        sample = F.sigmoid(self.net._decode(z_onehot)).data.transpose(-1,-2)
                    gifs.append(sample)
        '''
        # ============= Test for the image saver ==========
        val_table = [10,10,10,8,4,15]
        interpolation = torch.range(0,15,1)
        def cla_to_idx(cla):
            gap_table = [48000, 4800, 480, 60, 15,1]
            idx = 0
            for i in range(6):
                idx += cla[i]*gap_table[i]
            return idx
        
        img_vector = np.array([0,0,0,0,0,0])
        for kk in range(4):
            img_vector[4] = kk
            for i in range(6):
                for j in range(16):
                    if j < val_table[i]:
                        img_vector[i] = j
                    else:
                        img_vector[i] = val_table[i]-1
                    img_idx = cla_to_idx(img_vector)
                    
                    sample,_,_ = self.data_loader.dataset.__getitem__(img_idx)
                    sample = Variable(cuda(sample.float()/255, self.use_cuda), volatile=True).unsqueeze(0)
                    sample = sample.transpose(-1,-2).data
                    gifs.append(sample)
                
        # ============= End of Test for the image saver ==========
       '''
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

    def save_z_yc_pairs(self, out_z, out_yc):
        z_upk = unpack_batch_z(out_z)
        yc_upk = unpack_batch_y(out_yc)
        np.savez(self.metric_dir+'/zyc_pairs_gen'+str(self.global_gen)+'_it'+str(self.global_iter)+'.npz',
                 z = z_upk.cpu().numpy(),
                 yc = yc_upk.cpu().numpy()
                 )             
        
    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')
        if train:
            self.net.train()
        else:
            self.net.eval()
            
        if (self.model_type).lower()=='fvae':
            if train:
                self.D.train()
            else:
                self.D.eval()        

        
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
        
 
        
        
        



