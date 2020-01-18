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


z_values_dsprites = {}
z_values_dsprites['d0'] = [1., 2., 3.]
z_values_dsprites['d1'] = [0.5, 0.6, 0.7, 0.8, 0.9, 1. ]
z_values_dsprites['d2'] = [0.        , 0.16110732, 0.32221463, 0.48332195, 0.64442926,
   0.80553658, 0.96664389, 1.12775121, 1.28885852, 1.44996584,
   1.61107316, 1.77218047, 1.93328779, 2.0943951 , 2.25550242,
   2.41660973, 2.57771705, 2.73882436, 2.89993168, 3.061039  ,
   3.22214631, 3.38325363, 3.54436094, 3.70546826, 3.86657557,
   4.02768289, 4.1887902 , 4.34989752, 4.51100484, 4.67211215,
   4.83321947, 4.99432678, 5.1554341 , 5.31654141, 5.47764873,
   5.63875604, 5.79986336, 5.96097068, 6.12207799, 6.28318531]
z_values_dsprites['d3'] = [0.        , 0.03225806, 0.06451613, 0.09677419, 0.12903226,
   0.16129032, 0.19354839, 0.22580645, 0.25806452, 0.29032258,
   0.32258065, 0.35483871, 0.38709677, 0.41935484, 0.4516129 ,
   0.48387097, 0.51612903, 0.5483871 , 0.58064516, 0.61290323,
   0.64516129, 0.67741935, 0.70967742, 0.74193548, 0.77419355,
   0.80645161, 0.83870968, 0.87096774, 0.90322581, 0.93548387,
   0.96774194, 1.        ]
z_values_dsprites['d4'] = [0.        , 0.03225806, 0.06451613, 0.09677419, 0.12903226,
   0.16129032, 0.19354839, 0.22580645, 0.25806452, 0.29032258,
   0.32258065, 0.35483871, 0.38709677, 0.41935484, 0.4516129 ,
   0.48387097, 0.51612903, 0.5483871 , 0.58064516, 0.61290323,
   0.64516129, 0.67741935, 0.70967742, 0.74193548, 0.77419355,
   0.80645161, 0.83870968, 0.87096774, 0.90322581, 0.93548387,
   0.96774194, 1.        ]

z_values_3dshapes = {}
z_values_3dshapes['d0'] = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
z_values_3dshapes['d1'] = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
z_values_3dshapes['d2'] = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
z_values_3dshapes['d3'] = [0.75, 0.8214285714285714,  0.8928571428571428, 0.9642857142857143,
             1.0357142857142856, 1.1071428571428572, 1.1785714285714286, 1.25]
z_values_3dshapes['d4'] = [0., 1., 2., 3.]
z_values_3dshapes['d5'] = [-30., -25.71428571, -21.42857143, -17.14285714, -12.85714286,
                 -8.57142857, -4.28571429, 0., 4.28571429, 8.57142857, 12.85714286,
                 17.14285714, 21.42857143, 25.71428571, 30.]




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

    root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    data = np.load(root, allow_pickle=True, encoding='bytes')
    imgs = torch.from_numpy(data['imgs']).unsqueeze(1)
    vals = torch.from_numpy(data['latents_values'][:,1:]).unsqueeze(1)
    clas = torch.from_numpy(data['latents_classes'][:,1:]).unsqueeze(1)
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
    clas = torch.from_numpy(data['classes']).unsqueeze(1).long()
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
    
    
    