from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from mini_label_map import MINI_LABEL_MAP

class Mini_dataset(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the Pretrain_dataset dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.filenames = glob.glob(os.path.join(root, '*.jpg'))
        # print(self.filenames)

        self.len = len(self.filenames)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        image = Image.open(image_fn)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class Office_dataset(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the Pretrain_dataset dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        print("Loading all images to memory. This might take a while.")

        for filename in glob.glob(os.path.join(root, '*.jpg')):
            image = Image.open(filename)
            if self.transform is not None:
                image = self.transform(image)
            self.filenames.append( (image, MINI_LABEL_MAP[os.path.split(filename)[1][:-9]]) )
        # print(self.filenames)

        self.len = len(self.filenames)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        # image_fn, label = self.filenames[index]
        # image = Image.open(image_fn)
        # if self.transform is not None:
        #     image = self.transform(image)
        # #  print(image.shape)
        # return image, label
        return self.filenames[index]

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class Office_dataset_test(Dataset):
    def __init__(self, root, input_csv,  transform=None):
        self.filenames = []
        self.ids = []
        self.root = root
        self.transform = transform

        with open(input_csv, 'r') as f:
            for i, line in enumerate(f.readlines()):

                id, fn, _ = line.split(",")
                if i != 0: # Ignore first line
                    self.filenames.append(fn)
                    self.ids.append(id)

        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open( os.path.join(self.root, self.filenames[index]) )
        if self.transform is not None:
            image = self.transform(image)
        return image, self.filenames[index], self.ids[index]

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
