import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torchsummary import summary
from vgg16_aug import Vgg16_Aug
from dataset import P1_DATA

N_CLASS = 50 # number of classes
INPUT_SIZE = 224 # 128
BATCH_SIZE = 32
NUM_OF_WORKER = 8
CKPT_DIR = '../ckpt_p1/'
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)
DEVICE = 'cuda:0' # 'cuda:0', 'cpu' 
train_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ColorJitter(),
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

valid_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

# Use the torch dataloader to iterate through the dataset
trainset  = P1_DATA(root='../p1_data/train_50', transform=train_transform)
validset  = P1_DATA(root='../p1_data/val_50', transform=valid_transform)
trainset_loader  = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_OF_WORKER)
validset_loader  = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_OF_WORKER)

print('# images in trainset:', len(trainset)) # Should print 22500
print('# images in validset:', len(validset)) # Should print 2500

# get some random training images
dataiter = iter(trainset_loader)
images, labels = dataiter.next()
print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)

# Use GPU if available, otherwise stick with cpu
print("Cuda is available = " + str(torch.cuda.is_available()))
torch.manual_seed(123)
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device(DEVICE)
print('Device used:', device)
# print('torch.cuda.current_device() = ' + str(torch.cuda.current_device()))
# print('torch.cuda.get_device_name(0) = ' + str(torch.cuda.get_device_name(0)))

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

def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in validset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(validset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validset_loader.dataset),
        100. * correct / len(validset_loader.dataset)))

if __name__ == '__main__':
    vgg16_aug = Vgg16_Aug().to(device)
    # summary(vgg16_aug, (3, INPUT_SIZE, INPUT_SIZE))
    train_save(vgg16_aug, 100)
    pass
