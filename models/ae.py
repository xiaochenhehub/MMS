import torch.nn as nn
from pdb import set_trace
from .base_network import *

class Autoencoder(nn.Module):
    def __init__(self, nclass):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.reconstructor = Reconstructor()
        self.segmentor = Segmentor(nclass=nclass)

    def forward(self, x):
        f = self.encoder(x)
        rec, f_inner = self.reconstructor(f)
        seg = self.segmentor(f, f_inner)
        return rec, seg
    
    def get_f(self, x):
        return self.encoder(x)

    def get_rec_seg(self, f):
        rec, f_inner = self.reconstructor(f)
        seg = self.segmentor(f, f_inner)
        return rec, seg

def ae(nclass=5):
    return Autoencoder(nclass=nclass).cuda()
