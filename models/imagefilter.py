# GIN
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace


class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(self, out_channel = 32, in_channel = 3, scale_pool = [1, 3], layer_id = 0, use_act = True, requires_grad = False, **kwargs):
        super(GradlessGCReplayNonlinBlock, self).__init__()
        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.scale_pool     = scale_pool
        self.layer_id       = layer_id
        self.use_act        = use_act
        self.requires_grad  = requires_grad
        assert requires_grad == False

    def forward(self, x_in, params=None):
        nb, nc, nx, ny = x_in.shape
        x_in = x_in.view(1, nb * nc, nx, ny)

        if params is not None:
            idx_k = params[0]
            k = self.scale_pool[idx_k[0]]
            ker = params[1]
            shift = params[2]
        else:
            idx_k = torch.randint(high = len(self.scale_pool), size = (1,))
            k = self.scale_pool[idx_k[0]]
            ker = torch.randn([self.out_channel * nb, self.in_channel , k, k  ], requires_grad = self.requires_grad  ).cuda() 
            shift = torch.randn( [self.out_channel * nb, 1, 1 ], requires_grad = self.requires_grad  ).cuda() * 1.0  
            
        x_conv = F.conv2d(x_in, ker, stride =1, padding = k //2, dilation = 1, groups = nb )
        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)
        x_conv = x_conv.view(nb, self.out_channel, nx, ny)
        return x_conv, [idx_k, ker, shift]


class GINGroupConv(nn.Module):
    def __init__(self, out_channel = 3, in_channel = 3, interm_channel = 2, scale_pool = [1, 3 ], n_layer = 4, out_norm = 'frob', **kwargs):
        super(GINGroupConv, self).__init__()
        self.scale_pool = scale_pool # don't make it tool large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel

        self.layers.append(GradlessGCReplayNonlinBlock(out_channel = interm_channel, 
                                                       in_channel = in_channel, scale_pool = scale_pool, 
                                                       layer_id = 0).cuda())
        for ii in range(n_layer - 2):
            self.layers.append(GradlessGCReplayNonlinBlock(out_channel = interm_channel, 
                                                           in_channel = interm_channel, scale_pool = scale_pool, 
                                                           layer_id = ii + 1).cuda())
        self.layers.append(GradlessGCReplayNonlinBlock(out_channel = out_channel, 
                                                       in_channel = interm_channel, scale_pool = scale_pool, 
                                                       layer_id = n_layer - 1, use_act = False).cuda())
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x_in, alphas=None, params_list=None, return_parameter=False):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim = 0)
        nb, nc, nx, ny = x_in.shape
        # alpahs 
        if alphas is None:
            alphas = torch.rand(nb)[:, None, None, None] # nb, 1, 1, 1
            alphas = alphas.repeat(1, nc, 1, 1).cuda() # nb, nc, 1, 1
        # forward 
        if params_list is None:
            params_list = []
            x, params = self.layers[0](x_in)
            params_list.append(params)
            for blk in self.layers[1:]:
                x, params = blk(x)
                params_list.append(params)
        else:
            x, params = self.layers[0](x_in, params_list[0])
            layer_index = 1
            for blk in self.layers[1:]:
                x, params = blk(x, params_list[layer_index])
                layer_index += 1
        # mixed 
        mixed = alphas * x + (1.0 - alphas) * x_in
        # out norm
        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim = (-1, -2), p = 'fro', keepdim = False)
            _in_frob = _in_frob[:, None, None, None].repeat(1, nc, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
            _self_frob = _self_frob[:, None, None, None].repeat(1, self.out_channel, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5 ) ) * _in_frob
        if return_parameter:
            return mixed, alphas, params_list
        return mixed
