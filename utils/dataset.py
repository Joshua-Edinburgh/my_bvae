# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:42:00 2019

@author: Joshua

The dsprite-dataset, https://github.com/deepmind/dsprites-dataset

It has (737280 x 64 x 64, uint8) images in black and white, under key 'imgs'
It has (737280 x 6, float64) values of the latent factors, under key 'latents_values'
    1st: 1          Color, always 1.00
    2nd: 1,2,3      Shape, square, ellipse, heart
    3rd: 0.5~0.1    Scale, 6 values linearly between 0.5 and 1.0
    4th: 0.0~2pi    Rotation, 40 values linearly between 0 and 2pi
    5th: 0.0~1.0    Position X, 32 values between 0 and 1
    6th: 0.0~1.0    Position Y, 32 values between 0 and 1
"""

import os
import numpy as np
import h5py
import torch
import torch.utils.data as Data # Dataset, DataLoader



def return_data_dsprites(args):
    '''
        The initial version of data provider, no conditional sampling implemented
        Only feed all 737280 imgs and values to the data iterater.
    '''

    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    root = os.path.join(dset_dir, 'dsprites.npz')
    data = np.load(root, allow_pickle=True, encoding='bytes')
    imgs = torch.from_numpy(data['imgs']).unsqueeze(1)
    vals = torch.from_numpy(data['latents_values']).unsqueeze(1)
    clas = torch.from_numpy(data['latents_classes']).unsqueeze(1)
    data_set = Data.TensorDataset(imgs,vals,clas)

    train_loader = Data.DataLoader(
                              dataset = data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader





def return_data_3dshapes(args):    
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    root = os.path.join(dset_dir, '3dshapes.npz')
    data = np.load(root, allow_pickle=True, encoding='bytes')
    imgs = torch.from_numpy(data['images'])
    imgs = torch.transpose(imgs,1,3)
    vals = torch.from_numpy(data['values']).unsqueeze(1)
    clas = torch.from_numpy(data['classes']).unsqueeze(1)
    data_set = Data.TensorDataset(imgs,vals,clas)

    train_loader = Data.DataLoader(
                              dataset = data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader

if __name__ == '__main__':
    root = os.path.join('../'+args.dset_dir,'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dataset_zip = np.load(root, allow_pickle=True, encoding='bytes')
    # ======= Test whether the images can be correctly saved ========
    #ys_to_png_dsprite(out_y,args,dataset_zip)
    # ======= Test whether the ys correctly change to xbool_list ========
    #out_x = ys_to_xbool_dsprite(out_y,args,dataset_zip)
    
    
    