# Dataloader for abdominal images
import glob
import numpy as np
import dataloaders.niftiio as nio
import dataloaders.transform_utils as trans
import torch
import os
import platform
import torch.utils.data as torch_data
from functools import partial
import math
import itertools
from dataloaders.niftiio import read_nii_bysitk
from pdb import set_trace

def filter_samples_without_organ(img, lb):
    filtered_img = []
    filtered_lb = []
    for i in range(img.shape[0]):
        if np.any(lb[i] != 0):
            filtered_img.append(img[i])
            filtered_lb.append(lb[i])
    filtered_img = np.array(filtered_img)
    filtered_lb = np.array(filtered_lb)
    return filtered_img, filtered_lb

class mean_std_norm(object):
    def __init__(self,mean=None,std=None):
        self.mean=mean
        self.std=std

    def __call__(self,x_in):
        if self.mean is None:
            return (x_in-x_in.mean())/x_in.std()
        else:
            return (x_in-self.mean)/self.std

def get_normalize_op(fids, domain=False):
    def get_statistics(scan_fids):
        total_val = 0
        n_pix = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_val += in_img.sum()
            n_pix += np.prod(in_img.shape)
            del in_img
        meanval = total_val / n_pix

        total_var = 0
        for fid in scan_fids:
            in_img = read_nii_bysitk(fid)
            total_var += np.sum((in_img - meanval) ** 2 )
            del in_img
        var_all = total_var / n_pix

        global_std = var_all ** 0.5

        return meanval, global_std
    if not domain:
        return mean_std_norm()
    else:
        _mean, _std = get_statistics(fids)
        return mean_std_norm(_mean, _std)

hostname = platform.node()
# folder for datasets
BASEDIR = '/root/autodl-tmp/data/cardiac/processed'
print(f'Running on machine {hostname}, using dataset from {BASEDIR}')
LABEL_NAME = ["bg", "Myocardium",  "Lventricle","Rventricle"]

class CardiacDataset(torch_data.Dataset):
    def __init__(self,  mode, transforms, base_dir, domains: list, idx_pct = [0.7, 0.1, 0.2], 
                 tile_z_dim = 3, extern_norm_fn = None, filter_all_0=False):
        """
        Args:
            mode:               'train', 'val', 'test', 'test_all'
            transforms:         naive data augmentations used by default. Photometric transformations slightly better than those configured by Zhang et al. (bigaug)
            extern_norm_fn:     feeding data normalization functions from external, only concerns CT-MR cross domain scenario
        """
        super(CardiacDataset, self).__init__()
        self.transforms=transforms
        self.is_train = True if mode == 'train' else False
        self.phase = mode
        self.domains = domains
        self.all_label_names = LABEL_NAME
        self.nclass = len(LABEL_NAME)
        self.tile_z_dim = tile_z_dim
        self._base_dir = base_dir
        self.idx_pct = idx_pct
        self.filter_all_0 = filter_all_0

        self.img_pids = {}
        for _domain in self.domains: # load file names
            self.img_pids[_domain] = sorted([ fid.split("_")[-1].split(".nii.gz")[0] for fid in glob.glob(self._base_dir + "/" +  _domain  + "/image_*.nii.gz") ], key = lambda x: int(x))

        self.scan_ids = self.__get_scanids(mode, idx_pct) # train val test split in terms of patient ids

        self.info_by_scan = None
        self.sample_list = self.__search_samples(self.scan_ids) # image files names according to self.scan_ids
        if self.is_train:
            self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
        elif mode == 'test': # Source domain test
            self.pid_curr_load = self.scan_ids
        elif mode == 'test_all': # Choose this when being used as a target domain testing set. Liu et al.
            self.pid_curr_load = self.scan_ids

        self.normalize_op = extern_norm_fn([itm['img_fid'] for _, itm in self.sample_list[self.domains[0]].items()])

        print(f'For {self.phase} on {[_dm for _dm in self.domains]} using scan ids {self.pid_curr_load}')

        # load to memory
        self.actual_dataset = self.__read_dataset()
        self.size = len(self.actual_dataset) # 2D

    def __get_scanids(self, mode, idx_pct):
        """
        index by domains given that we might need to load multi-domain data
        idx_pct: [0.7 0.1 0.2] for train val test. with order te val tr
        """
        tr_ids      = {}
        val_ids     = {}
        te_ids      = {}
        te_all_ids  = {}

        for _domain in self.domains:
            dset_size   = len(self.img_pids[_domain])
            tr_size     = round(dset_size * idx_pct[0])
            val_size    = math.floor(dset_size * idx_pct[1])
            te_size     = dset_size - tr_size - val_size

            te_ids[_domain]     = self.img_pids[_domain][: te_size]
            val_ids[_domain]    = self.img_pids[_domain][te_size: te_size + val_size]
            tr_ids[_domain]     = self.img_pids[_domain][te_size + val_size: ]
            te_all_ids[_domain] = list(itertools.chain(tr_ids[_domain], te_ids[_domain], val_ids[_domain]   ))

        if self.phase == 'train':
            return tr_ids
        elif self.phase == 'val':
            return val_ids
        elif self.phase == 'test':
            return te_ids
        elif self.phase == 'test_all':
            return te_all_ids

    def __search_samples(self, scan_ids):
        """search for filenames for images and masks
        """
        out_list = {}
        for _domain, id_list in scan_ids.items():
            out_list[_domain] = {}
            for curr_id in id_list:
                curr_dict = {}

                _img_fid = os.path.join(self._base_dir, _domain   ,f'image_{curr_id}.nii.gz')
                _lb_fid  = os.path.join(self._base_dir, _domain   ,f'label_{curr_id}.nii.gz')

                curr_dict["img_fid"] = _img_fid
                curr_dict["lbs_fid"] = _lb_fid
                out_list[_domain][str(curr_id)] = curr_dict

        return out_list

    def __read_dataset(self):
        """
        Read the dataset into memory
        """
        out_list = []
        self.item_list = []
        self.info_by_scan = {} # meta data of each scan
        glb_idx = 0 # global index of a certain slice in a certain scan in entire dataset
        self.nframe_list = []

        for _domain, _sample_list in self.sample_list.items():
            for scan_id, itm in _sample_list.items():
                if scan_id not in self.pid_curr_load[_domain]:
                    continue

                img, _info = nio.read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out
                self.info_by_scan[_domain + '_' + scan_id] = _info

                img = np.float32(img)

                self.nframe_list.append(img.shape[0])

                if self.normalize_op.mean is not None:
                    vol_info={'vol_vmin':img.min(),'vol_vmax':img.max(),'vol_mean':self.normalize_op.mean,'vol_std':self.normalize_op.mean}
                else:
                    vol_info = {'vol_vmin': img.min(), 'vol_vmax': img.max(), 'vol_mean': img.mean(), 'vol_std': img.std()}
                img = self.normalize_op(img)

                lb = nio.read_nii_bysitk(itm["lbs_fid"])
                lb = np.float32(lb)

                if self.filter_all_0:
                    img, lb = filter_samples_without_organ(img, lb)

                self.item_list.append(
                    {'img': torch.from_numpy(img), "lb": torch.from_numpy(lb)}
                )

                img     = np.transpose(img, (1,2,0))
                lb      = np.transpose(lb, (1,2,0))

                assert img.shape[-1] == lb.shape[-1]

                # now start writing everthing in
                # write the beginning frame
                out_list.append( {"img": img[..., 0: 1],
                               "lb":lb[..., 0: 0 + 1],
                               "is_start": True,
                               "is_end": False,
                               "domain": _domain,
                               "nframe": img.shape[-1],
                               "scan_id": _domain + "_" + scan_id,
                               "z_id":0,
                               "vol_info":vol_info})
                glb_idx += 1

                for ii in range(1, img.shape[-1] - 1):
                    out_list.append( {"img": img[..., ii: ii + 1],
                               "lb":lb[..., ii: ii + 1],
                               "is_start": False,
                               "is_end": False,
                               "nframe": -1,
                               "domain": _domain,
                               "scan_id":_domain + "_" + scan_id,
                               "z_id": ii,
                               "vol_info":vol_info})
                    glb_idx += 1

                ii += 1 # last frame, note the is_end flag
                out_list.append( {"img": img[..., ii: ii + 1],
                               "lb":lb[..., ii: ii+ 1],
                               "is_start": False,
                               "is_end": True,
                               "nframe": -1,
                               "domain": _domain,
                               "scan_id":_domain + "_" + scan_id,
                               "z_id": ii,
                               "vol_info":vol_info})
                glb_idx += 1

        return out_list

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]
        if self.is_train:
            comp = np.concatenate([curr_dict["img"], curr_dict["lb"]], axis=-1)
            if self.transforms:
                img, lb = self.transforms(comp, c_img=1, c_label=1, 
                                          nclass=self.nclass, is_train=self.is_train, use_onehot=False)
            else:
                img, lb = curr_dict["img"], curr_dict["lb"]
            aug_img = 1
        else:
            img = curr_dict['img']
            lb = curr_dict['lb']
            aug_img = 1

        img = np.float32(img)
        lb = np.float32(lb)

        img = np.transpose(img, (2, 0, 1))
        lb  = np.transpose(lb, (2, 0, 1))

        img = torch.from_numpy( img )
        lb  = torch.from_numpy( lb )

        if self.tile_z_dim > 1:
            img = img.repeat( [ self.tile_z_dim, 1, 1] )
            assert img.ndimension() == 3

        is_start    = curr_dict["is_start"]
        is_end      = curr_dict["is_end"]
        nframe      = np.int32(curr_dict["nframe"])
        scan_id     = curr_dict["scan_id"]
        z_id        = curr_dict["z_id"]

        sample = {"img": img,
                "lb":lb.long(),
                "is_start": is_start,
                "is_end": is_end,
                "nframe": nframe,
                "scan_id": scan_id,
                "z_id": z_id,
                "aug_images": aug_img,
                }
        return sample

    def denorm_(self, img, vol_info):
        # scale to 0 - 1
        vmin, vmax, vmean, vstd = vol_info['vol_vmin'], vol_info['vol_vmax'], vol_info['vol_mean'], vol_info['vol_std']
        return ((img * vstd + vmean) - vmin) / (vmax - vmin)

    def renorm_(self, img, vol_info):
        vmin, vmax, vmean, vstd = vol_info['vol_vmin'], vol_info['vol_vmax'], vol_info['vol_mean'], vol_info['vol_std']
        return ((img * (vmax - vmin) + vmin) - vmean) / vstd

    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        return len(self.actual_dataset)

tr_func  = trans.transform_with_label(trans.tr_aug)

def get_training(modality,  tile_z_dim = 3, filter_all_0=False):
    return CardiacDataset(
        mode = 'train',\
        domains = modality,\
        transforms = None,\
        base_dir = BASEDIR,\
        extern_norm_fn = partial(get_normalize_op,domain=True),
        tile_z_dim = tile_z_dim,
        filter_all_0=filter_all_0)

def get_validation(modality,  tile_z_dim = 3):
     return CardiacDataset(
        mode = 'val',\
        transforms = None,\
        domains = modality,\
        base_dir = BASEDIR,\
        extern_norm_fn =partial(get_normalize_op,domain=False),
        tile_z_dim = tile_z_dim)

def get_test(modality,  tile_z_dim = 3):
     return CardiacDataset(
        mode = 'test',\
        transforms = None,\
        domains = modality,\
        extern_norm_fn = partial(get_normalize_op,domain=False),
        base_dir = BASEDIR,\
        tile_z_dim = tile_z_dim)

