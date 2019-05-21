
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

EPOCH = 15
BATCH_SIZE = 64
n = 2  # num_workers
LATENT_CODE_NUM = 64
log_interval = 10

class DetNet(nn.Module):
    def __init__(self):
        super(DetNet, self).__init__()


    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)  # batch_s, 8, 7, 7
        mu = self.fc11(out1.view(out1.size(0), -1))  # batch_s, latent
        logvar = self.fc12(out2.view(out2.size(0), -1))  # batch_s, latent
        z = self.reparameterize(mu, logvar)  # batch_s, latent
        out3 = self.fc2(z).view(z.size(0), 256, 16, 16)  # batch_s, 8, 7, 7

        return self.decoder(out3), mu, logvar


def loss_func(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD