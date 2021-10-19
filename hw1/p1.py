import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image

N_CLASS = 50 # number of classes

class P1_DATA(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the P1_DATA dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(N_CLASS):
            filenames = glob.glob(os.path.join(root, '1_' + '*.png'))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

# Load dataset
trainset = P1_DATA(root='p1_data/train_50', transform=transforms.ToTensor())
testset  = P1_DATA(root='p1_data/val_50', transform=transforms.ToTensor())

print('# images in trainset:', len(trainset)) # Should print 22500
print('# images in testset:', len(testset)) # Should print 2500

# Use the torch dataloader to iterate through the dataset
trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)
testset_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)

# get some random training images
dataiter = iter(trainset_loader)
images, labels = dataiter.next()

print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# model = Net().to(device) # Remember to move the model to "device"
# print(model)





