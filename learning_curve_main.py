import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from solver import Solver, reconstruction_loss
from utils.basic import str2bool
from torch.autograd import Variable
from utils.basic import cuda
from model import reparametrize
from tqdm import tqdm
import sys
import torch.utils.data as Data # Dataset, DataLoader
from metrics import Metric_R

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def show_images_grid(imgs_, num_images=25):
  ncols = int(np.ceil(num_images**0.5))
  nrows = int(np.ceil(num_images / ncols))
  _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
  axes = axes.flatten()

  for ax_i, ax in enumerate(axes):
    if ax_i < num_images:
      ax.imshow(imgs_[ax_i], cmap='Greys_r',  interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
    else:
      ax.axis('off')

def show_density(imgs):
  _, ax = plt.subplots()
  ax.imshow(imgs.mean(axis=0), interpolation='nearest', cmap='Greys_r')
  ax.grid('off')
  ax.set_xticks([])
  ax.set_yticks([])

def smooth_x(x,ratio=10):
    if type(x) != np.ndarray:
        x = np.asarray(x)
    new_x = np.zeros(x.shape)
    tmp = x[0]
    for i in range(x.size):
        tmp = (1-1/ratio)*tmp + 1/ratio*x[i]
        new_x[i] = tmp
    return new_x


def prepare_dataset(args,file_name):
    data_file = np.load('data/'+file_name)
    all_x = data_file['out_x']
    all_y = data_file['out_y']
    all_fz = data_file['out_fz']

    perm_table = np.arange(0,all_fz.shape[0],1)
    np.random.shuffle(perm_table)
    perm_y, perm_fz = all_y[perm_table], all_fz[perm_table]
    
    imgs = torch.from_numpy(all_x).unsqueeze(1)
    vals = torch.from_numpy(all_y).unsqueeze(1)
    fulz = torch.from_numpy(all_fz).unsqueeze(1)
    data_set = Data.TensorDataset(imgs,vals,fulz)

    orig_loader = Data.DataLoader(
                              dataset = data_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)

    perm_vals = torch.from_numpy(perm_y).unsqueeze(1)
    perm_fulz = torch.from_numpy(perm_fz).unsqueeze(1)
    perm_data_set = Data.TensorDataset(imgs, perm_vals, perm_fulz)

    perm_loader = Data.DataLoader(
                              dataset = perm_data_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)
    metric = Metric_R(args)
    dist_orig, comp_orig, info_orig, R_orig = metric.dise_comp_info([fulz[:6400,:,:args.z_dim]],[vals[:6400,:,:]],'random_forest')
    dist_perm, comp_perm, info_perm, R_perm = metric.dise_comp_info([perm_fulz[:6400,:,:args.z_dim]],[vals[:6400,:,:]],'random_forest')
    return orig_loader, perm_loader, dist_orig[-1], dist_perm[-1]


def decoder_curves(args, data_loader, flag=False):    
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)
    net.net_mode(True)
    out = False
    
    pre_DE_cnt = 0
    pbar = tqdm(total=net.niter_preDE)
    pbar.update(pre_DE_cnt)
    loss_table = []
    
    while not out:
        for x, y, mz in data_loader:
            x = Variable(cuda(x.float(), net.use_cuda))
            y = Variable(cuda(y.float(), net.use_cuda))
            mz = Variable(cuda(mz.float(), net.use_cuda)) 
            #z = mz[:,:,:args.z_dim]
            z = reparametrize(mz[:,:,:args.z_dim], mz[:,:,args.z_dim:])
            if flag:
                y_zeros = torch.zeros_like(y)
                z = torch.cat((y,y_zeros),dim=2)       
            x_recon = net.net._decode(z) 
            loss = reconstruction_loss(x, x_recon, 'bernoulli')
            loss_table.append(loss.data.item())
            
            net.optim_DE.zero_grad()
            loss.backward()
            net.optim_DE.step()
            
            if pre_DE_cnt >= net.niter_preDE:
                out = True
                break
            pre_DE_cnt += 1
            pbar.update(1)
            
    pbar.write("[Pretrain Encoder Finished]") 
    pbar.close() 
    sys.stdout.flush() 
    return loss_table

def encoder_curves(args,data_loader,flag=False):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)
    net.net_mode(True)
    out = False
    
    pre_EN_cnt = 0
    pbar = tqdm(total=net.niter_preEN)
    pbar.update(pre_EN_cnt)
    loss_fun = torch.nn.MSELoss()
    loss_table = []
    while not out:
        for x, y, mz in data_loader:
            x = Variable(cuda(x.float(), net.use_cuda))
            y = Variable(cuda(y.float(), net.use_cuda))
            mz = Variable(cuda(mz.float(), net.use_cuda))
            mz_bar = mz[:,:,:args.z_dim]
            mz_hat = net.net._encode(x)[:,:args.z_dim]
            #mz_bar = reparametrize(mz[:,:,:args.z_dim], mz[:,:,args.z_dim:])
            #tmp = net.net._encode(x)
            #mz_hat = reparametrize(tmp[:,:args.z_dim], tmp[:,args.z_dim:])
            if flag:
                y_zeros = torch.zeros_like(y)
                mz_bar = torch.cat((y,y_zeros),dim=2)
            loss = loss_fun(mz_bar,mz_hat)
            loss_table.append(loss.data.item())
            
            net.optim_EN.zero_grad()
            loss.backward()
            net.optim_EN.step()
              
            if pre_EN_cnt >= net.niter_preEN:
                out = True
                break
            pre_EN_cnt += 1
            pbar.update(1)
    pbar.write("[Pretrain Encoder Finished]") 
    pbar.close() 
    sys.stdout.flush()  

    return loss_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1343432, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter_per_gen', default=1000, type=int, help='maximum training iteration per generation')
    parser.add_argument('--max_gen', default=10, type=int, help='number of generations')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--a_dim', default=40, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--nb_preENDE', default=100, type=int, help='Number of batches for pre-train encoder and decoder')
    parser.add_argument('--niter_preEN', default=1000, type=int, help='Number of max iterations for pre-train encoder')
    parser.add_argument('--niter_preDE', default=1000, type=int, help='Number of max iterations for pre-train decoder')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')
    
    parser.add_argument('--save_step', default=1e4, type=int, help='number of iterations after which a checkpoint is saved')
    parser.add_argument('--metric_step',default=1e4, type=int, help='number of iterations after which R and top_sim metric saved')
    parser.add_argument('--top_sim_batches',default=1000,type=int, help='number of batches of sampling z when calculating top_sim and R')
    parser.add_argument('--save_gifs',default=True, type=str2bool, help='whether save the gifs')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--exp_name', default='learning_curves', type=str, help='name of the experiment')
    
    args = parser.parse_args()
        
    data_loader1,perm_loader1, dist_o1, dist_p1 = prepare_dataset(args,'saved_xyz_550000.npz')
    data_loader2,perm_loader2, dist_o2, dist_p2 = prepare_dataset(args,'dis_high.npz')
#    loss_table1 = decoder_curves(args,data_loader1,True)
#    loss_table2 = decoder_curves(args,perm_loader1,True)
    loss_table1 = encoder_curves(args,data_loader1,True)
    loss_table2 = encoder_curves(args,perm_loader1,True)
    
    x_axis = np.arange(0,len(loss_table1),1)
    plt.plot(x_axis,smooth_x(loss_table1,50),'r',label=str(dist_o1))
    plt.plot(x_axis,smooth_x(loss_table2,50),'b',label=str(dist_p1))
    plt.legend()
    plt.grid(True)
    plt.show()