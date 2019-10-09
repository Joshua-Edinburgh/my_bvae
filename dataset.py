# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:42:00 2019

@author: Joshua
"""

import os
import numpy as np

import torch
import torch.utils.data as Data # Dataset, DataLoader


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0



def return_data(args):
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    data = np.load(root, allow_pickle=True, encoding='bytes')
    imgs = torch.from_numpy(data['imgs']).unsqueeze(1)
# =================

    train_loader = Data.DataLoader(
                              dataset = imgs,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader

test =return_data(args)
flag = 0
for x in test:
    print(x)
    flag += 1
    if flag >2:
        break