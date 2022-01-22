# This is from https://github.com/lucidrains/byol-pytorch

import torchvision.transforms as transforms
import torch
from byol_pytorch import BYOL
import torchvision
from dataset import Mini_dataset
from torch.utils.data import DataLoader
import torch.nn as nn

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)
INPUT_SIZE = 128
BATCH_SIZE = 32 # 128
NUM_OF_WORKER = 8
CKPT_DIR = '../p2_ckpt/'
DEVICE = "cuda"

# This is defined by TA
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    # Data augumentation
    # transforms.ColorJitter(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=10),
    # 
    transforms.ToTensor(),
    transforms.Normalize(mean = NORM_MEAN, std = NORM_STD)])

# Use the torch dataloader to iterate through the dataset
trainset  = Mini_dataset(root='../hw4_data/mini/train', transform=train_transform)
trainset_loader  = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_OF_WORKER)
print('# images in trainset:', len(trainset)) # Should print 

# get some random training images
dataiter = iter(trainset_loader)
images = dataiter.next()
print('Image tensor in each batch:', images.shape, images.dtype)

# Use GPU if available, otherwise stick with cpu
print("Cuda is available = " + str(torch.cuda.is_available()))
torch.manual_seed(123)


model = torchvision.models.resnet50(pretrained=False) # TODO fine-tune the MLP layers
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 65)
)

model = model.to(DEVICE)
learner = BYOL(
    model,
    image_size = INPUT_SIZE,
    hidden_layer = 'avgpool',
)

def train(model, epoch, log_interval=100):
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    for ep in range(epoch):
        iteration = 0
        for batch_idx, data in enumerate(trainset_loader):

            data = data.to(DEVICE)
            loss = learner(data)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        

        torch.save(model.state_dict(), CKPT_DIR + f'C-{ep}.pth')
        print(f"Saved model to {CKPT_DIR + f'C-{ep}.pth'}")
    
    # # save the final model
    # save_checkpoint(f'B-{ep}.pth', model, optimizer)
    # save your improved network
    torch.save(model.state_dict(), CKPT_DIR + f'C-{ep}.pth')
    print(f"Saved model to {CKPT_DIR + f'C-{ep}.pth'}")

if __name__ == '__main__':
    train(model, 40) # 100