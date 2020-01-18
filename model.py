"""model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils.basic import cuda
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical



def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

# ======================== BetaVAE_H ==========================================
class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=5, nc=3):
        super(BetaVAE_H, self).__init__()    
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
                
    def encoder_init(self):
        for m in self._modules['encoder']:
            kaiming_init(m)
    
    def decoder_init(self):
        for m in self._modules['decoder']:
            kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar, z.squeeze()

    def fd_gen_z(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        z = mu
        return z.data

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

# ======================== Categorical VAE ====================================
class CVAE(nn.Module):
    """Catogorical VAE structure
    """

    def __init__(self, z_dim=5, nc=3,a_dim=40,gumbel_tmp=1.0):
        super(CVAE, self).__init__()
        self.gumbel_tmp = gumbel_tmp
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),
            nn.Linear(256, z_dim*a_dim)             # B, z_dim*2
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim*a_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
                
    def encoder_init(self):
        for m in self._modules['encoder']:
                kaiming_init(m)
                
    def decoder_init(self):
        for m in self._modules['decoder']:
            kaiming_init(m)

    def forward(self, x):
        hidden = self._encode(x)
        hidden_matrix = hidden.view(-1,self.z_dim,self.a_dim)
        sftmx = F.softmax(hidden_matrix,dim=-1)
        z_matrix = RelaxedOneHotCategorical(self.gumbel_tmp, probs=sftmx).rsample()
        x_recon = self._decode(z_matrix.view(-1,self.z_dim*self.a_dim))
        z_argmx = OneHotCategorical(probs=sftmx).sample().argmax(dim=-1)
        return x_recon, sftmx, z_argmx.squeeze().float()
        

    def fd_gen_z(self, x):
        hidden = self._encode(x)
        hidden_matrix = hidden.view(-1,self.z_dim,self.a_dim)
        sftmx = F.softmax(hidden_matrix,dim=-1)
        z_argmx = OneHotCategorical(probs=sftmx).sample().argmax(dim=-1)
        return z_argmx
        

    def _encode(self, x):
        return self.encoder(x)            # B, 256

    def _decode(self, z_matrix):
        return self.decoder(z_matrix)


# ======================== Factor VAE =========================================
class F_Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(F_Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 2),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()


class FVAE(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, z_dim=5, nc=3):
        super(FVAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 2*z_dim, 1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim, 128, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.nc, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x, no_dec=False):
        stats = self._encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            x_recon = self._decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()

    def fd_gen_z(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        z = mu
        return z.data

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class FCVAE(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, z_dim=15, nc=3, a_dim=40,gumbel_tmp=2.0):
        super(FCVAE, self).__init__()
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.gumbel_tmp = gumbel_tmp
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            nn.ReLU(True),
            View((-1,128*1*1)),
            nn.Linear(128,z_dim*a_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim*a_dim, 128),
            View((-1,128,1,1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.nc, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x, no_dec=False):
        hidden = self._encode(x)
        hidden_matrix = hidden.view(-1,self.z_dim,self.a_dim)
        sftmx = F.softmax(hidden_matrix,dim=-1)
        z_matrix = RelaxedOneHotCategorical(self.gumbel_tmp, probs=sftmx).rsample()
        x_recon = self._decode(z_matrix.view(-1,self.z_dim*self.a_dim))  
        z_argmx = OneHotCategorical(probs=sftmx).sample().argmax(dim=-1)
        if no_dec:
            return z_argmx.squeeze().float()
        else:
            return x_recon, sftmx, z_argmx.squeeze().float()
        
    def fd_gen_z(self, x):
        hidden = self._encode(x)
        hidden_matrix = hidden.view(-1,self.z_dim,self.a_dim)
        sftmx = F.softmax(hidden_matrix,dim=-1)
        z_argmx = sftmx.argmax(dim=-1)
        return z_argmx.float()

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)



# ========================== VQ VAE ===========================================
class VQVAE(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""
    def __init__(self, z_dim=5, nc=3, a_dim=40, nz = 1):
        super(VQVAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.a_dim = a_dim
        self.nz = nz
        self.encoder = nn.Sequential(
            nn.Conv2d(self.nc, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(128, nz*z_dim, 1)
        )
        
        self.embd = nn.Embedding(self.a_dim,self.z_dim).cuda()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(nz*z_dim, 128, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.nc, 4, 2, 1)
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            if block != 'embd':
                for m in self._modules[block]:
                    initializer(m)
                
    def find_nearest(self,query,target):
        Q=query.unsqueeze(1).repeat(1,target.size(0),1)
        T=target.unsqueeze(0).repeat(query.size(0),1,1)
        index=(Q-T).pow(2).sum(2).sqrt().min(1)[1]
        return target[index]

    def hook(self, grad):
        self.grad_for_encoder = grad
        return grad
   
    def forward(self, x):
        z_enc = self._encode(x).squeeze(-1).squeeze(-1)
        z_dec = self.find_nearest(z_enc, self.embd.weight)
        z_dec.register_hook(self.hook)
        x_recon = self._decode(z_dec.view(-1,self.z_dim,1,1))
        z_enc_for_embd = self.find_nearest(self.embd.weight,z_enc)
        return x_recon, z_enc, z_dec, z_enc_for_embd

    def fd_gen_z(self, x):
        z_enc = self._encode(x).squeeze(-1).squeeze(-1)
        z_dec = self.find_nearest(z_enc, self.embd.weight)
        return z_dec.data

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

if __name__ == '__main__':
    net = CVAE()
    net.weight_init()



'''
class IVAE(nn.Module):
    """Our VAE structure, with mu(X) discrete.
        The mapping matrix W must feeded in cuda() type
    """

    def __init__(self, W, z_dim=10, nc=3,a_dim=40):
        super(IVAE, self).__init__()
        self.gumble_idx = 2.0
        self.W = W
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1))
            #nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        
        self.encoder_mu = nn.Sequential(
            nn.Linear(256, z_dim*(a_dim))       
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(256, z_dim)       
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
                
    def encoder_init(self):
        for m in self._modules['encoder']:
                kaiming_init(m)
        for m in self._modules['encoder_logvar']:
                kaiming_init(m)
        for m in self._modules['encoder_mu']:
                kaiming_init(m)
                
    def decoder_init(self):
        for m in self._modules['decoder']:
            kaiming_init(m)

    def forward(self, x):

        long_embed = self._encode(x)                # B, 256
        logvar = self.encoder_logvar(long_embed)    # B, z_dim
        tmp = self.encoder_mu(long_embed)           # B, (a_dim)*z_dim
        for i in range(self.z_dim):
            lp, rp = (self.a_dim)*i, (self.a_dim)*(i+1)
            sfmx_tmp = F.softmax(tmp[:,lp:rp],dim=1)         # B, a_dim
            gumb_tmp = RelaxedOneHotCategorical(self.gumble_idx, probs=sfmx_tmp).rsample()
            mu_tmp = torch.mm(gumb_tmp,self.W.unsqueeze(1))  # B, 1
            if i==0:
                mu = mu_tmp
            else:
                mu = torch.cat((mu,mu_tmp),dim=1)   # Finally, it is [B, z_dim]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)           
        return x_recon, mu, logvar
        

    def fd_gen_z(self, x):
        """Whether sample from distribution or directly return mu?"""
        long_embed = self._encode(x)                # B, 256
        #logvar = self.encoder_logvar(long_embed)    # B, z_dim
        tmp = self.encoder_mu(long_embed)           # B, (a_dim)*z_dim
        for i in range(self.z_dim):
            lp, rp = (self.a_dim)*i, (self.a_dim)*(i+1)
            sfmx_tmp = F.softmax(tmp[:,lp:rp],dim=1)         # B, a_dim
            gumb_tmp = RelaxedOneHotCategorical(self.gumble_idx, probs=sfmx_tmp).rsample()
            mu_tmp = torch.mm(gumb_tmp,self.W.unsqueeze(1))  # B, 1
            if i==0:
                mu = mu_tmp
            else:
                mu = torch.cat((mu,mu_tmp),dim=1)   # Finally, it is [B, z_dim]
        #z = reparametrize(mu, logvar)         
        return mu.data
        

    def _encode(self, x):
        return self.encoder(x)            # B, 256

    def _decode(self, z):
        return self.decoder(z)
'''