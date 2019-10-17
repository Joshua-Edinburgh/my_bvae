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
from matplotlib import pyplot as plt

import torch
import torch.utils.data as Data # Dataset, DataLoader


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0

# Helper function to show images
def save_images_grid(imgs_, args, num_images=25, idx_figs=1):
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    fig, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    filename = 'figures_'+str(idx_figs)+'.pdf'
    file_path = os.path.join('exp_results/'+args.exp_name+'/images')
    if not os.path.exists(file_path):
       os.makedirs(file_path) 
    fig.savefig(os.path.join(file_path, filename))
    
def show_density(imgs):
    _, ax = plt.subplots()
    ax.imshow(imgs.mean(axis=0), interpolation='nearest', cmap='Greys_r')
    ax.grid('off')
    ax.set_xticks([])
    ax.set_yticks([])

def return_data(args):
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
    vals = torch.from_numpy(data['latents_values']).unsqueeze(1)
    data_set = Data.TensorDataset(imgs,vals)

    train_loader = Data.DataLoader(
                              dataset = data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader

def y_to_xidx_dsprite(y):
    b_size = y.shape[0]
    # ====== Fill y back to B*6 vector =================
    if y.shape[-1] < 6:
        fill = torch.ones((b_size,6-y.shape[-1]),dtype=y.dtype)
        y_fill = torch.cat((fill,y),dim=1).squeeze(0)
    else:
        y_fill = y
    # ====== Translate y back to indicies
    y_fill[:,0:1] = torch.ones((b_size,1),dtype=y.dtype)*0.
    y_fill[:,1:2] = torch.ones((b_size,1),dtype=y.dtype)*(y_fill[:,1:2]-1.)
    y_fill[:,2:3] = torch.ones((b_size,1),dtype=y.dtype)*(y_fill[:,2:3]*10.-5.)
    y_fill[:,3:4] = torch.ones((b_size,1),dtype=y.dtype)*(np.round(y_fill[:,3:4]/(2.*np.pi/40.)))    
    y_fill[:,4:5] = torch.ones((b_size,1),dtype=y.dtype)*(np.round(y_fill[:,4:5]*32.))
    y_fill[:,5:6] = torch.ones((b_size,1),dtype=y.dtype)*(np.round(y_fill[:,5:6]*32.))
    
    latents_sizes = np.asarray([ 1,  3,  6, 40, 32, 32])
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],np.array([1,])))
    x_idx = np.dot(y_fill, latents_bases).astype(int)
    x_idx[np.where(x_idx>737279)] = 737279
    # ===== x_idx shape with (B,)
    return x_idx
  
def ys_to_xidxs_dsprite(y_list):
    x_idx_list = []
    for y in y_list:
        x_idx_list.append(y_to_xidx_dsprite(y))
    # ======= Each element in x_idx_list has shape (B,)
    return x_idx_list
    
def ys_to_png_dsprite(y_list,args,dataset_zip):
    x_idx_list = ys_to_xidxs_dsprite(y_list)
    all_imgs = dataset_zip['imgs']  
    for i, batch_imgs in enumerate(x_idx_list):
        save_images_grid(all_imgs[batch_imgs], args, num_images=args.batch_size, idx_figs=i)

def ys_to_xbool_dsprite(y_list,args,dataset_zip):
    x_idx_list = ys_to_xidxs_dsprite(y_list)
    all_imgs = dataset_zip['imgs']  
    x_list = []
    for i, batch_imgs in enumerate(x_idx_list):
        x_list.append(all_imgs[batch_imgs])
    return x_list

if __name__ == '__main__':
    root = os.path.join('../'+args.dset_dir,'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dataset_zip = np.load(root, allow_pickle=True, encoding='bytes')
    # ======= Test whether the images can be correctly saved ========
    #ys_to_png_dsprite(out_y,args,dataset_zip)
    # ======= Test whether the ys correctly change to xbool_list ========
    out_x = ys_to_xbool_dsprite(out_y,args,dataset_zip)
    
    
    