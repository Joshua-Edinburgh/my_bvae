#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:38:26 2020

@author: joshua
"""

import os
import numpy as np
import h5py
import torch
import torch.utils.data as Data # Dataset, DataLoader

def vals_to_clas(vals):
# ================ Values to classes =======================
    clas = np.zeros_like(vals)  
    for i in range(clas.shape[1]):
        table = np.sort(list(set(vals[:,i])))
        for j in range(len(table)):
            mask = vals[:,i]==table[j]
            clas[mask,i] = j
    return clas

root = os.path.join('../data', '3dshapes.h5')
h5File = h5py.File(root, 'r')
imgs = np.array(h5File['images'])
vals = np.array(h5File['labels'])
clas = vals_to_clas(vals)

np.savez('../data/3dshapes.npz',
         images = imgs,
         values = vals,
         classes = clas)

