import torch
import torch.utils.data as data


class PairDataset(data.Dataset):
    def __init__(self, img, lb):
        self.img = img
        self.lb = lb

    def __getitem__(self, index):
        item = {"img":self.img[index], 
                "lb": self.lb[index]}
        return item

    def __len__(self):
        return len(self.img)
