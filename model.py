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


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, W, z_dim=10, nc=3, a_dim=40):
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

        return x_recon, mu, logvar

    def fd_gen_z(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        z = mu
        return z.data

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

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
        '''
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar
        '''
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


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    net = IVAE()
    net.weight_init()
