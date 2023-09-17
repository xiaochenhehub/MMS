import sys
import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import my_utils.util as util
from my_utils.util import *
from .base_model import *
from pdb import set_trace
import numpy as np
import random
import matplotlib.pyplot as plt
from .ae import ae
import torch.nn.functional as F
from .loss import *


class Net(BaseModel):

    def set_networks(self, nclass=5):
        self.model = ae(nclass=nclass)

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.alpha = opt.alpha
        self.dataset_max = torch.tensor(opt.dataset_max).cuda()
        self.dataset_min = torch.tensor(opt.dataset_min).cuda()
        self.nclass = opt.nclass
        self.batchsize = opt.batchsize
        self.set_networks(self.nclass)
        self.criterionRec = RecLoss().cuda()
        self.criterionDice = SoftDiceLoss(self.nclass).cuda()
        self.criterionWCE = My_CE(nclass = self.nclass, 
                                  batch_size = self.batchsize, 
                                  weight = torch.ones(self.nclass,)).cuda()
        self.criterionMSE = nn.MSELoss().cuda()
        self.ScoreDiceEval = Efficient_DiceScore(self.nclass, ignore_chan0 = True).cuda()
        self.optimizers = []
        self.schedulers = []
        self.optimizer = torch.optim.Adam( itertools.chain(self.model.parameters()), lr=opt.lr)
        self.optimizer_seg = torch.optim.Adam( itertools.chain(self.model.segmentor.parameters()), lr=opt.lr)
        self.optimizers.append(self.optimizer)
        self.optimizers.append(self.optimizer_seg)
        for i in range(len(self.optimizers)):
            self.schedulers.append(get_scheduler(self.optimizers[i], opt))
        if opt.continue_train:
            if self.opt.data_name == "PROSTATE":
                reload_model_dir = opt.reload_model_dir.replace("*", opt.tr_domain)
            else:
                reload_model_dir = opt.reload_model_dir
            self.model.encoder.load_state_dict(torch.load(reload_model_dir))
            self.model.reconstructor.load_state_dict(torch.load(reload_model_dir.replace("ENC", "REC")))
            # self.model.segmentor.load_state_dict(torch.load(opt.reload_model_dir.replace("ENC", "SEG")))
            print(f"continue train at dir {reload_model_dir}")
        self.loss_rec_norm = 0
        self.loss_seg = 0
        self.loss_f_consist = 0
        self.dices = 0

    def set_input(self, input):
        self.input_img = input['img'].cuda()
        self.input_mask = input['lb'].cuda()
        self.img_norm, self.input_img_max, self.input_img_min = norm_each_slice(self.input_img)

    def set_input_pair(self, input):
        img_pair, lb_pair = input['img'].cuda(), input['lb'].cuda()
        img, img_refer = img_pair[:, :3, :, :], img_pair[:, 3:, :, :]
        lb, lb_refer = lb_pair[:, :1, :, :], lb_pair[:, 1:, :, :]
        self.input_img = img
        self.input_mask = lb
        self.img_norm, self.input_img_max, self.input_img_min = norm_each_slice(img)
        self.img_refer_norm, self.input_img_refer_max, self.input_img_refer_min = norm_each_slice(img_refer)
    
    def forward_rec(self):
        rec_norm, _ = self.model(self.img_norm)
        rec = denorm(rec_norm, self.input_img_max, self.input_img_min)
        loss_rec_norm = self.criterionRec(rec_norm, self.img_norm)
        loss_rec = self.criterionRec(rec, self.input_img)
        self.loss = loss_rec
        self.loss_rec_norm = loss_rec_norm.data
        self.loss_rec = loss_rec.data

    def forward_inter(self):
        # rec
        f_norm = self.model.get_f(self.img_norm)
        rec_norm, _ = self.model.get_rec_seg(f_norm)
        # inter 
        lam = np.random.beta(self.alpha, self.alpha, size=self.img_norm.shape[0])
        lam = torch.tensor(lam, dtype=torch.float32).view(self.img_norm.shape[0], 1, 1, 1).cuda()
        rec = denorm(rec_norm, self.input_img_max, self.input_img_min)
        f_refer_norm = self.model.get_f(self.img_refer_norm)
        f_inter = f_norm + lam * (f_refer_norm - f_norm)
        rec_inter_norm, _ = self.model.get_rec_seg(f_inter)
        re_f_inter = self.model.get_f(rec_inter_norm)
        # loss
        loss_rec_norm = self.criterionRec(rec_norm, self.img_norm)
        loss_rec = self.criterionRec(rec, self.input_img)
        loss_f_consist = self.criterionMSE(f_inter, re_f_inter)
        self.loss = loss_rec + loss_f_consist * 1000
        self.loss_rec = loss_rec.data
        self.loss_rec_norm = loss_rec_norm.data
        self.loss_f_consist = loss_f_consist.data

    def forward_seg(self):
        rec_norm, seg = self.model(self.img_norm)
        
        rec = denorm(rec_norm, self.input_img_max, self.input_img_min)

        loss_dice = self.criterionDice(seg, self.input_mask)
        loss_wce = self.criterionWCE(seg, self.input_mask.long())
        loss_seg = loss_dice + loss_wce
        self.loss = loss_seg

        self.loss_rec_norm = self.criterionRec(rec_norm, self.img_norm).data
        self.loss_rec = self.criterionRec(rec, self.input_img).data
        self.loss_seg = loss_seg.data
        self.dices = self.ScoreDiceEval(torch.argmax(seg, 1, keepdim=True), 
                                        torch.squeeze(self.input_mask, 1), 
                                        dense_input = True).mean().data

    def optimize_parameters_rec(self):
        self.forward_rec()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def optimize_parameters_inter(self):
        self.forward_inter()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def optimize_parameters_seg(self):
        self.forward_seg()
        self.optimizer_seg.zero_grad()
        self.loss.backward()
        self.optimizer_seg.step()
