import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
from torchsummary import summary

N_CLASS = 7 # number of classes
INPUT_SIZE = 512
BATCH_SIZE = 32
CKPT_DIR = 'ckpt_p2/'

# Clear ckpt directory TODO


class P2_DATA(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the P2_DATA dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(N_CLASS):
            filenames = glob.glob(os.path.join(root, str(i) + '_' + '*.png'))
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

train_transform = transforms.Compose([
    transforms.ToTensor()
])

valid_transform = transforms.Compose([
    transforms.ToTensor()
])

# Use the torch dataloader to iterate through the dataset
trainset = P1_DATA(root='p2_data/train', transform=train_transform)
testset  = P1_DATA(root='p2_data/validation', transform=valid_transform)
trainset_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
testset_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

print('# images in trainset:', len(trainset)) # Should print 22500
print('# images in testset:', len(testset)) # Should print 2500

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

def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in testset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))

def train(model, epoch, log_interval=100):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()  # Important: set training mode
    
    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
            
        test(model) # Evaluate at the end of each epoch

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def train_save(model, epoch, log_interval=100):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()  # set training mode
    
    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        
        test(model)
        
        if ep % 10 == 0:
            save_checkpoint(CKPT_DIR + 'p1-%i.pth' % ep, model, optimizer)
    
    # save the final model
    save_checkpoint('p1-%i.pth' % ep, model, optimizer)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vgg16_feature = models.vgg16(pretrained=True).features
        self.transpose_conv = nn.Sequential(
        )

    def forward(self, x):
        x = self.vgg16_feature(x)
        x = self.transpose_conv(x)
        return x

vgg16_aug = Net().to(device)
summary(vgg16_aug, (3, INPUT_SIZE, INPUT_SIZE))
train_save(vgg16_aug, 100)