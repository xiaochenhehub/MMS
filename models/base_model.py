import os
import torch
import pdb
from torch.optim import lr_scheduler
from pdb import set_trace

# helper functions
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class BaseModel(object):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

    def set_input(self, input):
        self.input_img = input['img'].cuda()
        self.input_mask = input['lb'].cuda()

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def lr_update(self):
        for _scheduler in self.schedulers:
            _scheduler.step()

    def save_network(self, network, dir, epoch):
        os.makedirs(dir, exist_ok=True)
        file_name = f"net_{epoch}.pth"
        dir = os.path.join(dir, file_name)
        torch.save(network.cpu().state_dict(), dir)
        network.cuda()

    def load_network(self, network, dir):
        network.load_state_dict(torch.load(dir))
        print(f'Load: network {dir} as been loaded')

    def save_optimizer(self,optimizer, dir, epoch):
        os.makedirs(dir, exist_ok=True)
        file_name = f"optim_{epoch}.pth"
        dir = os.path.join(dir, file_name)
        torch.save(optimizer.state_dict(), dir)

    def load_optimizer(self, optimizer, dir):
        optimizer.load_state_dict(torch.load(dir))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
    
    def show_learning_rate(self):
        for optimizer in self.optimizers:
            lr = optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
