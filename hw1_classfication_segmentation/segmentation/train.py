import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import FCN32s, UNet, FCN8s
from dataset import P2_DATA

BATCH_SIZE = 4 # Don't use 8
DEVICE = 'cuda:1'
NUM_OF_WORKER = 0
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE = 512
CKPT_DIR = '../ckpt_p2/'

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

# Load dataset
print("loading training dataset.....")
trainset = P2_DATA(root='../p2_data/train/',      transform=train_transform)
# validset = P2_DATA(root='../p2_data/validation/', transform=valid_transform)
print("Complete image loading")

trainset_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_OF_WORKER)
# validset_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_OF_WORKER)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device(DEVICE)
print('Device used:', device)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def train_fcn32s(model, epoch, log_interval=100):
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    model.train()  # set training mode

    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1

        if ep % 10 == 0:
            save_checkpoint(CKPT_DIR + 'p2-%i.pth' % ep, model, optimizer)

    # save the final model
    save_checkpoint('p2-%i.pth' % ep, model, optimizer)

fcn8s = FCN8s().to(device)
train_fcn32s(fcn8s, 100, log_interval=10)
