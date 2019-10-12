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

import torch
import torch.nn as nn
import pandas as pd
import numpy as np


# ====== For these distance functions, x1 and x2 should be vector with size [x]
def cos_dist(x1,x2):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    return cos(x1,x2)

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



class Metrics:
    def __init__(self,args):
        self.b_siz = args.batch_size
        self.smp_flag = True   # When z,y is too large, use True, we may sample
        self.smp_size = 10000     # The number of sampled pairs
        self.z_dist = 'cosine'
        self.y_dist = 'cosine'
        self.x_dist = 'cosine'
        
    def unpack_batch_zy(self,z_list,y_list):
        z_dim = z_list[0].shape[-1]
        y_dim = y_list[0].shape[-1]
        z_upk = torch.stack(z_list).view(-1,z_dim)
        y_upk = torch.stack(y_list).view(-1,y_dim)
        return z_upk, y_upk
        
    def tensor_dist(self,tens1,tens2,dist_type='cosine'):
        if dist_type == 'cosine':
            return cos_dist(tens1,tens2)
        if dist_type == 'edit':
            return edit_dist(tens1,tens2)
        else:
            raise NotImplementedError
        
        
    def top_sim_zy(self, z_list, y_list):
        z_upk, y_upk = self.unpack_batch_zy(z_list, y_list)
        smp_cnt = self.smp_size
        len_zy = z_upk.shape[0]
        z_dist = []
        y_dist = []    
        
        if self.smp_flag:  
            smp_set_list = []
            while smp_cnt > 0:
                i,j = np.random.randint(0,len_zy,size=2)
                smp_set_list = set([i,j])
                if set([i,j]) not in smp_set_list:
                    smp_cnt -= 1
                    z_dist.append(self.tensor_dist(z_upk[i],z_upk[j],self.z_dist))
                    y_dist.append(self.tensor_dist(y_upk[i],y_upk[j],self.y_dist))                               
        else:
            for i in range(len_zy):
                for j in range(i):
                    if i!=j:
                        z_dist.append(self.tensor_dist(z_upk[i],z_upk[j],self.z_dist))
                        y_dist.append(self.tensor_dist(y_upk[i],y_upk[j],self.y_dist))
                        
        dist_table = pd.DataFrame({'ZD':np.asarray(z_dist),
                                   'YD':np.asarray(y_dist)})
        corr_pearson = dist_table.corr()['ZD']['YD']
            
        return corr_pearson

#metric_test = Metrics(args)
#corr = metric_test.top_sim_zy(out_z,out_y)
