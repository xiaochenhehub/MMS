import time
import SimpleITK as sitk
import numpy as np
import dataloaders.niftiio as nio

from models import create_forward
from my_utils.util import AttrDict, worker_init_fn

from torch.utils.data import DataLoader
from tqdm import tqdm
from configs_exp import ex
import dataloaders.AbdominalDataset as ABD
import dataloaders.ProstateDataset as PROS
import dataloaders.CardiacDataset as CAR
from pdb import set_trace
import os
from dataloaders.PairDataset import PairDataset as PD
import torch

@ex.automain
def main(_run, _config, _log):
    # config 
    opt = AttrDict(_config)
    
    # dataset
    if opt.data_name == "ABDOMINAL":
        train_set = ABD.get_training(modality = [opt.tr_domain] )
    elif opt.data_name == 'PROSTATE':
        train_set       = PROS.get_training(modality = opt.tr_domain, filter_all_0=opt.filter_all_0)
    elif opt.data_name == "CARDIAC":
        train_set = CAR.get_training(modality = [opt.tr_domain])

    print(f'Using TR domain {opt.tr_domain}; TE domain {opt.te_domain}')
    train_loader = DataLoader(dataset = train_set, num_workers = 8,
                              batch_size = opt.batchsize, shuffle = opt.shuffle, drop_last = True, 
                              worker_init_fn = worker_init_fn, pin_memory = True)
    opt.dataset_max, opt.dataset_min = 0, 0
    
    # inter dataset
    refer_shift = opt.refer_shift
    img_list = []
    lb_list = []
    for train_batch in train_loader:
        img_list.append(train_batch['img'])
        lb_list.append(train_batch['lb'])
    imgs = torch.concat(img_list, dim=0)
    lbs = torch.concat(lb_list, dim=0)
    batch_size = imgs.shape[0]
    distances = torch.cdist(imgs.view(batch_size, -1), imgs.view(batch_size, -1))
    refer_indices = torch.argsort(distances, dim=1)[:, refer_shift]
    imgs_refer = imgs[refer_indices]
    lbs_refer = lbs[refer_indices]

    img_pair = torch.concat([imgs, imgs_refer], dim=1)
    lb_pair = torch.concat([lbs, lbs_refer], dim=1)

    pair_dataset = PD(img_pair, lb_pair)
    pair_dataloader = DataLoader(pair_dataset, num_workers=8, drop_last=True, 
                                batch_size= 20, shuffle=True, pin_memory=True)
    if opt.train_type == "inter":
        loader = pair_dataloader
    else:
        loader = train_loader
    
    # trainer
    trainer = create_forward(opt)

    # trian process
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        np.random.seed()
        epoch_log, epoch_loss_rec, epoch_loss_rec_norm, epoch_loss_seg = 0, 0, 0, 0
        epoch_dices = 0
        epoch_loss_f_consist = 0
        for train_batch in tqdm(loader, total=len(loader)):
            train_input = {'img': train_batch["img"], 'lb': train_batch["lb"]}
            if opt.train_type == "inter":
                trainer.set_input_pair(train_input)
            else:
                trainer.set_input(train_input)
            if opt.train_type == "rec":
                trainer.optimize_parameters_rec()
            elif opt.train_type == "inter":
                trainer.optimize_parameters_inter()
            elif opt.train_type == "seg":
                trainer.optimize_parameters_seg()
            epoch_log += 1
            epoch_loss_rec += trainer.loss_rec
            epoch_loss_rec_norm += trainer.loss_rec_norm
            epoch_loss_seg += trainer.loss_seg
            epoch_dices += trainer.dices
            epoch_loss_f_consist += trainer.loss_f_consist
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        print(f"loss_rec: {epoch_loss_rec/epoch_log:.4f}")
        print(f"loss_rec_norm: {epoch_loss_rec_norm/epoch_log:.4f}")
        print(f"loss_seg: {epoch_loss_seg/epoch_log:.4f}")
        print(f"dices: {epoch_dices/epoch_log:.4f}")
        print(f"f_consist: {epoch_loss_f_consist/epoch_log:.4f}")
        if epoch % opt.save_epoch_freq == 0:
            trainer.save_network(trainer.model.encoder, os.path.join(opt.save_dir, opt.tr_domain, "ENC"), epoch)
            trainer.save_network(trainer.model.reconstructor, os.path.join(opt.save_dir, opt.tr_domain, "REC"), epoch)
            trainer.save_network(trainer.model.segmentor, os.path.join(opt.save_dir, opt.tr_domain, "SEG"), epoch)
            print(f"save network at dir {opt.save_dir} in epoch {epoch}")
        trainer.show_learning_rate()
        if epoch > opt.niter:
            trainer.update_learning_rate()
