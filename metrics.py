#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 22:56:03 2019

Provide two quantitive metrics:
    1. Topological similarity (by Simon in xxxx)
    2. R matrix (by Cian in xxxx)

The input should be:
    out_z: List, each element has size B*z_dim
    out_y: List, each element has size B*6
@author: xiayezi
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble.forest import RandomForestRegressor
from utils.hinton import hinton
import matplotlib.pyplot as plt
from utils.dataset import ys_to_xbool_dsprite

# ====== For these distance functions, x1 and x2 should be vector with size [x]
def cos_dist(x1,x2):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    return cos(x1,x2)

def euclidean_dist(x1,x2,p=2):
    return torch.dist(x1,x2,p=p)

def edit_dist(x1,x2):
    len1, len2 = x1.shape[0], x2.shape[0]
    DM = [0]
    for i in range(len1):
        DM.append(i+1)
        
    for j in range(len2):
        DM_new=[j+1]
        for i in range(len1):
            tmp = 0 if x1[i]==x2[j] else 1
            new = min(DM[i+1]+1, DM_new[i]+1, DM[i]+tmp)
            DM_new.append(new)
        DM = DM_new
        
    return DM[-1]


def mse(predicted, target):
    ''' mean square error '''
    predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted #(n,)->(n,1)
    target = target[:, None] if len(target.shape) == 1 else target #(n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0] #value not array

def rmse(predicted, target):
    ''' root mean square error '''
    return np.sqrt(mse(predicted, target))

def nmse(predicted, target):
    ''' normalized mean square error '''
    return mse(predicted, target) / np.var(target)

def nrmse(predicted, target):
    ''' normalized root mean square error '''
    return rmse(predicted, target) / np.std(target)

def unpack_batch_zoy(zoy_list):
    zoy_dim = zoy_list[0].shape[-1]
    zoy_upk = torch.stack(zoy_list).view(-1,zoy_dim)
    return zoy_upk

def unpack_batch_x(x_list):
    x_tensor_list = []
    for x in x_list:
        x_tensor_list.append(torch.from_numpy(x))
    return torch.stack(x_tensor_list).view(-1,64,64)

class Metric_R:
    def __init__(self,args):
        self.b_siz = args.batch_size
        self.y_dim = 5
        self.z_dim = args.z_dim
        self.seed = args.seed
        self.metric_dir = os.path.join('exp_results/'+args.exp_name+'/metrics')
        if not os.path.exists(self.metric_dir):
            os.makedirs(self.metric_dir)
        
    def norm_entropy(self,p):
        '''p: probabilities '''
        n = p.shape[0]
        return - p.dot(np.log(p + 1e-12) / np.log(n + 1e-12))

    def entropic_scores(self, r):
        '''r: relative importances '''
        r = np.abs(r)
        ps = r / np.sum(r, axis=0) # 'probabilities'
        hs = [1-self.norm_entropy(p) for p in ps.T]
        return hs

    def fit_get_scores(self, z_list, y_list, reg_model, reg_paras, err_fn, attr):
        z_upk = unpack_batch_zoy(z_list)  # x_upk has shape B*x_dim
        y_upk = unpack_batch_zoy(y_list)
        if z_upk.is_cuda:
            z_upk = z_upk.cpu()
        if y_upk.is_cuda:
            y_upk = y_upk.cpu()        
        R = []
        errs = np.zeros((self.y_dim+1))     # The last store the average
        
        for i in range(self.y_dim):
            model = reg_model(**reg_paras[i])
            model.fit(z_upk,y_upk[:,i])     # X is all z, y is one colunm
            
            y_pred = model.predict(z_upk)
            errs[i] = err_fn(y_pred, y_upk[:,i].numpy())
            r = getattr(model, attr)[:,None]    # Should be z_dim*1
            R.append(np.abs(r))
        R = np.hstack(R)
        # =========== cal. disentanglement ===============
        c_rel_importance = np.sum(R,1) / np.sum(R) # relative importance of each code variable
        disent_scores = self.entropic_scores(R.T)
        disent_w_avg = np.sum(np.array(disent_scores) * c_rel_importance)
        disent_scores.append(disent_w_avg)          # [z_dim+1, 1]
        # =========== cal. completeness ==================
        complete_scores = self.entropic_scores(R)
        complete_avg = np.mean(complete_scores)
        complete_scores.append(complete_avg)       # [y_dim+1, 1]
        # =========== cal. informativeness ===============
        errs[-1] = np.mean(errs[:-1])
        info_scores = errs
        
        return disent_scores, complete_scores, info_scores, R
    
    def dise_comp_info(self, z_list, y_list, reg_model='lasso', show_fig=False):
        
        if reg_model.lower() == 'lasso':
            reg_paras = [{"alpha": 0.02}] * self.y_dim         
            disent_scores, complete_scores, info_scores, R = \
                self.fit_get_scores(z_list,y_list,Lasso,reg_paras,nrmse,'coef_')
        elif reg_model.lower() == 'random_forest':
            reg_paras = []
            rng = np.random.RandomState(self.seed)
            y_max_depths = [12, 10, 3, 3, 3]
            for y_max_depth in y_max_depths:
                reg_paras.append({"n_estimators":self.z_dim, 
                                  "max_depth":y_max_depth, 
                                  "random_state": rng})
            disent_scores, complete_scores, info_scores, R = \
                self.fit_get_scores(z_list,y_list,RandomForestRegressor,
                                    reg_paras,nrmse,'feature_importances_')              
        else:
            raise('Only "lasso or random_forest"')
          
        return disent_scores, complete_scores, info_scores, R
    
    def hinton_fig(self,R_list, model_list):
        num_figures = len(R_list)
        siz_x = int(num_figures*4)
        fig, axes = plt.subplots(1,num_figures,figsize=(siz_x,12))
        axes = axes.ravel()
        part_name = ''
        for i, R in enumerate(R_list):           
            hinton(R, '$\mathbf{z}$', '$\mathbf{c}$',ax=axes[i], fontsize=18)  
            axes[i].set_title('{0}'.format(model_list[i]), fontsize=20)
            part_name += '_'
            part_name += model_list[i].lower()            
        fig_name = self.metric_dir+'/hinton'+part_name+'.pdf'
        fig.savefig(fig_name)



class Metric_topsim:
    def __init__(self,args):
        self.b_siz = args.batch_size
        self.smp_flag = False   # When z,y is too large, use True, we may sample
        self.smp_size = 1e5     # The number of sampled pairs
        self.z_metr = 'EU'
        self.y_metr = 'EU'
        self.x_metr = 'XEU'
        
    def tensor_dist(self,tens1,tens2,dist_type='cosine'):
        if dist_type == 'cosine':
            return cos_dist(tens1,tens2)
        if dist_type == 'edit':
            return edit_dist(tens1,tens2)
        if dist_type == 'EU':
            return euclidean_dist(tens1,tens2,p=2)
        if dist_type == 'XEU':
            return euclidean_dist(tens1.view(-1).float(),tens2.view(-1).float(),p=2)
        if dist_type == 'Xcosine':
            return cos_dist(tens1.view(-1).float(),tens2.view(-1).float())
        else:
            raise NotImplementedError
        
    def top_sim_zy(self, z_list, y_list):
        z_upk = unpack_batch_zoy(z_list)  # x_upk has shape B*x_dim
        y_upk = unpack_batch_zoy(y_list)
        if z_upk.is_cuda:
            z_upk = z_upk.cpu()
        if y_upk.is_cuda:
            y_upk = y_upk.cpu()
        len_zy = z_upk.shape[0]
        smp_cnt = self.smp_size
        smp_left = (len_zy**2-len_zy)*0.2     
        z_dist = []
        y_dist = []    
        
        if self.smp_flag:  
#            smp_set_list = []
            while smp_cnt > 0 and smp_left > 0:
                i,j = np.random.randint(0,len_zy,size=2)
#                if set([i,j]) not in smp_set_list and i!=j:
#                    smp_set_list.append(set([i,j]))
#                    smp_left -= 1
                if i!=j:
                    smp_cnt -= 1
                    z_dist.append(self.tensor_dist(z_upk[i],z_upk[j],self.z_metr))
                    y_dist.append(self.tensor_dist(y_upk[i],y_upk[j],self.y_metr))
        else:
            for i in range(len_zy):
                for j in range(i):
                    if i!=j:
                        z_dist.append(self.tensor_dist(z_upk[i],z_upk[j],self.z_metr))
                        y_dist.append(self.tensor_dist(y_upk[i],y_upk[j],self.y_metr))            
        dist_table = pd.DataFrame({'ZD':np.asarray(z_dist),
                                   'YD':np.asarray(y_dist)})
        corr_pearson = dist_table.corr()['ZD']['YD']            
        return corr_pearson
    
    def top_sim_xzoy(self, zoy_list, x_list):
        zoy_upk = unpack_batch_zoy(zoy_list)        # To [lis*b_size,zoy_dim]
        x_upk = unpack_batch_x(x_list)               # To [lis*b_size,64,64]
        if zoy_upk.is_cuda:
            zoy_upk = zoy_upk.cpu()
        if x_upk.is_cuda:
            x_upk = x_upk.cpu()   
        len_x = x_upk.shape[0]
        smp_cnt = self.smp_size
        smp_left = (len_x**2-len_x)*0.2    
        zoy_dist = []
        x_dist = []
        
        if self.smp_flag:  
#            smp_set_list = []
            while smp_cnt > 0 and smp_left > 0:
                i,j = np.random.randint(0,len_x,size=2)
#                if set([i,j]) not in smp_set_list and i!=j:
#                    smp_set_list.append(set([i,j]))
#                    smp_left -= 1
                if i!=j:
                    smp_cnt -= 1
                    zoy_dist.append(self.tensor_dist(zoy_upk[i],zoy_upk[j],self.z_metr))
                    x_dist.append(self.tensor_dist(x_upk[i],x_upk[j],self.x_metr))                        
        else:
            for i in range(len_x):
                for j in range(i):
                    if i!=j:
                        zoy_dist.append(self.tensor_dist(zoy_upk[i],zoy_upk[j],self.z_metr))
                        x_dist.append(self.tensor_dist(x_upk[i],x_upk[j],self.x_metr))
        print('dis_xzoy_done')               
        dist_table = pd.DataFrame({'ZOYD':np.asarray(zoy_dist),
                                   'XD':np.asarray(x_dist)})
        corr_pearson = dist_table.corr()['ZOYD']['XD']            
        return corr_pearson

if __name__ == "__main__":

    # ========== Test for Metric_topsim ================
    # ====== Must run dataset.py to prepare variable: dataset_zip
    #metric_topsim = Metric_topsim(args)
    #corr = metric_topsim.top_sim_zy(out_z,out_y)
    #print('Topsim between z and y is: %.4f'%corr)
    #out_x = ys_to_xbool_dsprite(out_y,args,dataset_zip)
    #corrX_Y = metric_topsim.top_sim_xzoy(out_y,out_x)
    #print('Topsim between y and X is: %.4f'%corrX_Y)
    # ========== Test for Metric_R ================
    metric_R = Metric_R(args)
    a1, b1, c1, R1 = metric_R.dise_comp_info(out_z,out_y,'lasso')
    #a2, b2, c2, R2 = metric_R.dise_comp_info(out_z,out_y,'random_forest')
    #model_list = ['Base-Lasso','Base-RF']
    #metric_R.hinton_fig([R1,R2], model_list)
    
    