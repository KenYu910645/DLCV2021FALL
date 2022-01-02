import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from dataset import Office_dataset_test
from mini_label_map import MINI_LABEL_MAP_INV

INPUT_SIZE = 128
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device used:', DEVICE)

def load_checkpoint(model, ckpt_path):
    state = torch.load(ckpt_path)
    model.load_state_dict(state['state_dict'])

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_csv', type=str, default='../hw4_data/val.csv')
    parser.add_argument('--img_dir', type=str, default='./hw4_data/val')
    parser.add_argument('--output_csv', type=str, default='output/val_pred.csv')
    parser.add_argument('--ckpt_path',   type=str, default='../hw4_p2_best.pth')
    config = parser.parse_args()
    print(config)

    # This is defined by TA
    valid_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean = NORM_MEAN, std = NORM_STD)])

    # Use the torch dataloader to iterate through the dataset
    validset  = Office_dataset_test(root=config.img_dir, input_csv = config.img_csv , transform=valid_transform)
    validset_loader  = DataLoader(validset, batch_size=1, shuffle=False)

    # Model init
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 65)
    )
    load_checkpoint(model, config.ckpt_path)
    print(f'model loaded from {config.ckpt_path}')

    model = model.to(DEVICE)

    # Predict
    model.eval()
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        s = "id,filename,label\n"
        for data, fn, id in validset_loader:
            data = data.to(DEVICE)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            s += f"{id[0]},{fn[0]},{MINI_LABEL_MAP_INV[pred.item()]}\n"

    with open(config.output_csv, 'w') as f:
        f.write(s)