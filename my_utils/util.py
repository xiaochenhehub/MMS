from typing import Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def recursive_init(cls, myDict):
        """
        recursively look into a dictionary and convert each sub_dictionary entry to AttrDict
        This is a little bit messy
        """
        def _rec_into_subdict(curr_dict):
            for key, entry in curr_dict.items():
                if type(entry) is dict:
                    _rec_into_subdict(entry)
                    curr_dict[key] = cls(entry)
        _rec_into_subdict(myDict)
        return cls(myDict)
    

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


def filter_samples_without_organ_torch(img, lb):
    filtered_img = []
    filtered_lb = []
    for i in range(img.shape[0]):
        if torch.any(lb[i] != 0):
            filtered_img.append(img[i])
            filtered_lb.append(lb[i])
    filtered_img = torch.stack(filtered_img)
    filtered_lb = torch.stack(filtered_lb)
    return filtered_img, filtered_lb


def dice_score(segmentation, lb, num_classes):
    scores = []
    for i in range(segmentation.size(0)):  # Iterate over batch dimension
        seg_item = segmentation[i]
        lb_item = lb[i]
        item_scores = []
        for class_idx in range(1, num_classes):
            seg_class = seg_item == class_idx
            lb_class = lb_item == class_idx
            intersection = torch.sum(seg_class * lb_class)
            union = torch.sum(seg_class) + torch.sum(lb_class)
            dice = (2 * intersection) / (union + 1e-8)
            item_scores.append(dice)
        scores.append(item_scores)
    return scores


class Normalizer:
    def __init__(self, dataset_max, dataset_min):
        self.dataset_max = dataset_max
        self.dataste_min = dataset_min

    def __call__(self, tensor) -> torch.Tensor:
        return (tensor - self.dataste_min) / (self.dataset_max - self.dataste_min)


class Denormalizer:
    def __init__(self, dataset_max, dataset_min):
        self.dataset_max = dataset_max
        self.dataste_min = dataset_min

    def __call__(self, tensor) -> torch.Tensor:
        return tensor * (self.dataset_max - self.dataste_min) + self.dataste_min


def norm(tensor, max, min):
    return (tensor - min) / (max - min)

def norm_each_slice(tensor):
    max = torch.amax(tensor, dim=(2, 3), keepdim=True)
    min = torch.amin(tensor, dim=(2, 3), keepdim=True)    
    same_values_mask = max == min
    max[same_values_mask] += 1e-5
    min[same_values_mask] -= 1e-5
    norm_tensor = (tensor - min) / (max - min)
    return norm_tensor, max, min

def denorm(tensor, max, min):
    return tensor * (max - min) + min

def t2i(tensor, detach=True):
    if isinstance(tensor, np.ndarray):
        return np.transpose(tensor, [1, 2, 0])
    if detach:
        return torch.squeeze(tensor).detach().cpu()
    else:
        return torch.squeeze(tensor).cpu()


def vis_img(img, figsize=4, cmap="gray", transform=True, detach=True):
    if img.ndim == 3:
        img = img[0]
    if transform:
        img = t2i(img, detach=detach)
    f = plt.figure(figsize=(figsize, figsize))
    plt.imshow(img, cmap=cmap)
    plt.axis("off")

def vis_lb(img, vmax, figsize=4, transform=True, detach=True):
    cmap = plt.cm.get_cmap("rainbow")
    norm = mcolors.Normalize(vmin=1, vmax=vmax)
    if img.ndim == 3:
        img = img[0]
    if transform:
        img = t2i(img, detach=detach)
    f = plt.figure(figsize=(figsize, figsize))
    img = np.ma.masked_where(img == 0, img)
    plt.imshow(img, cmap, norm)
    plt.axis("off")

def vis_img_list(imgs:torch.Tensor, figsize=4, transform=True, detach=True, cmap="gray"):
    num = len(imgs)
    f = plt.figure(figsize=(figsize, num*figsize))
    for i in range(num):
        f.add_subplot(1, num, i+1)
        img = imgs[i]
        if img.ndim == 3:
            img = img[0]
        if transform:
            img = t2i(img, detach=detach)
        plt.imshow(img, cmap=cmap)
        plt.axis("off")

def vis_pair_list(imgs, lbs, title=None, figsize=4, transform=True, detach=True, vmin=0, vmax=4):
    num = len(imgs)
    f = plt.figure(figsize=(num*figsize, 2*figsize))
    for i in range(num):
        # show img
        f.add_subplot(2, num, i+1)
        img = imgs[i]
        if img.ndim == 3:
            img = img[0]
        if transform:
            img = t2i(img, detach=detach)
        plt.imshow(img, cmap='gray')
        if title:
            plt.title(title[i], fontsize=23)
        plt.axis("off")
        # show label 
        f.add_subplot(2, num, i+1+num)
        if transform:
            label = t2i(lbs[i], detach=detach)
        plt.imshow(label, vmin=vmin, vmax=vmax)
        plt.axis("off")

def vis_pair_list_overlap(imgs, lbs, title=None, figsize=4, 
                          transform=True, detach=True, vmin=1, vmax=4):
    num = len(imgs)
    cmap = plt.cm.get_cmap("rainbow")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    f = plt.figure(figsize=(num*figsize, figsize))
    for i in range(num):
        # show img
        f.add_subplot(1, num, i+1)
        img = imgs[i]
        if img.ndim == 3:
            img = img[0]
        if transform:
            img = t2i(img, detach=detach)
        plt.imshow(img, cmap='gray')
        # show label 
        if transform:
            lb = t2i(lbs[i], detach=detach)
        mask = np.ma.masked_where(lb == 0, lb)
        plt.imshow(mask, cmap=cmap, alpha=0.5, norm=norm)
        if title:
            plt.title(title[i], fontsize=23)
        plt.axis("off")

def vis_pair_overlap(img, lb, figsize=4, transform=True, 
                     detach=True, vmin=1, vmax=4, title=None):
    cmap = plt.cm.get_cmap("rainbow")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    f = plt.figure(figsize=(figsize, figsize))
    # show img
    if transform:
        img = t2i(img, detach=detach)
    plt.imshow(img, cmap='gray')
    # show label 
    if transform:
        lb = t2i(lb, detach=detach)
    mask = np.ma.masked_where(lb == 0, lb)
    if title:
        plt.title(title)
    plt.imshow(mask, cmap=cmap, alpha=0.5, norm=norm)
    plt.axis("off")