from torch.utils.data import Dataset
import glob
import os
import torch
from PIL import Image
import copy # Avoid PIL compliant
import torchvision.transforms as transforms
N_CLASS = 7

def mask_target(im): # TODO WTF
    im = transforms.ToTensor()(im)
    im = 4 * im[0] + 2 * im[1] + 1 * im[2]
    target = torch.zeros(im.shape, dtype=torch.long)
    target[im==3] = 0
    target[im==6] = 1
    target[im==5] = 2
    target[im==2] = 3
    target[im==1] = 4
    target[im==7] = 5
    target[im==0] = 6
    target[im==4] = 6
            
    return target

class P2_DATA(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the P2_DATA dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.sate_img = []
        self.mask_img = []

        # read filenames
        sate_filenames = glob.glob(os.path.join(root, '*.jpg'))
        # mask_filenames = glob.glob(os.path.join(root, '*.png'))
        for fn in sate_filenames:
            self.filenames.append((fn, fn[:-7] + 'mask.png')) # (sate_filename, mask_filename) pair
        self.len = len(self.filenames)

        # Load image to memory
        for fn in self.filenames:
            self.sate_img.append(copy.deepcopy(Image.open(fn[0])))
            self.mask_img.append(copy.deepcopy(Image.open(fn[1])))

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        sate_img = self.sate_img[index]
        mask_img = self.mask_img[index]

        if self.transform is not None:
            sate_img = self.transform(sate_img)
        return sate_img, mask_target(mask_img)

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len