import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from dataset import Office_dataset

INPUT_SIZE = 128
BATCH_SIZE = 32
NUM_OF_WORKER = 8
CKPT_DIR = '../p2_ckpt_tmp/'
FREEZE_BACKBONE = False
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)
DEVICE = 'cuda' # 'cuda:0', 'cpu' # TODO

# This is defined by TA
train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean = NORM_MEAN, std = NORM_STD)])

# Use the torch dataloader to iterate through the dataset
trainset  = Office_dataset(root='../hw4_data/office/train', transform=train_transform)
validset  = Office_dataset(root='../hw4_data/office/val',   transform=train_transform)
trainset_loader  = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_OF_WORKER)
validset_loader  = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_OF_WORKER)

print('# images in trainset:', len(trainset)) # Should print 3951
print('# images in validset:', len(validset)) # Should print 406

# get some random training images
dataiter = iter(trainset_loader)
images, labels = dataiter.next()
print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)

# Use GPU if available, otherwise stick with cpu
print("Cuda is available = " + str(torch.cuda.is_available()))
torch.manual_seed(123)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def train(model, epoch, log_interval=100):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()  # set training mode
    
    iteration = 0
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(trainset_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
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
        save_checkpoint(CKPT_DIR + f'{ep}.pth', model, optimizer)
    
    # save the final model
    save_checkpoint(f'{ep}.pth', model, optimizer)


def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in validset_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(validset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validset_loader.dataset),
        100. * correct / len(validset_loader.dataset)))

def load_checkpoint(model, ckpt_path):
    state = torch.load(ckpt_path)
    model.load_state_dict(state['state_dict'])

if __name__ == '__main__':
    
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 65)
    )
    # Load model weight
    # TA_RESNET50_CKPT_PATH = "../hw4_data/pretrain_model_SL.pt"
    # model.load_state_dict(torch.load(TA_RESNET50_CKPT_PATH))
    # print(f'model loaded from {TA_RESNET50_CKPT_PATH}')
    TA_RESNET50_CKPT_PATH = "../p2_ckpt/C-20.pth"
    model.load_state_dict(torch.load(TA_RESNET50_CKPT_PATH))
    print(f'model loaded from {TA_RESNET50_CKPT_PATH}')

    model = model.to(DEVICE)

    # Freezee backbone
    if FREEZE_BACKBONE:
        for ct, child in enumerate(model.children()):
            if ct < 9:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print("Only unfrozen below layers")
                print(child)

   
    train(model, 100)