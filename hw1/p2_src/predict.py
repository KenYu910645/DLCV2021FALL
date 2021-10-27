import os
import argparse
import glob
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model import FCN32s, UNet, FCN8s
from PIL import Image
from shutil import rmtree

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)
EVAL = True # If you wanna use mean_iou_evaluate.py to evaluate

MASK = {
    0: (0, 1, 1),
    1: (1, 1, 0),
    2: (1, 0, 1),
    3: (0, 1, 0),
    4: (0, 0, 1),
    5: (1, 1, 1),
    6: (0, 0, 0),
}

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='../p2_data/validation/')
    parser.add_argument('--output_path', type=str, default='output/')
    parser.add_argument('--model', default='FCN8s', type=str)
    parser.add_argument('--ckpt_path', default='../ckpt_p2_fcn32/p2-40.pth', type=str)
    config = parser.parse_args()
    print(config)

    # 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    # Get device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device used:', device)

    # Load model
    if config.model == 'FCN8s':
        model = FCN8s().to(device)
    elif config.model == 'UNet':
        model = UNet().to(device)
    else:
        model = FCN32s().to(device)
    state = torch.load(config.ckpt_path)
    model.load_state_dict(state['state_dict'])

    # save directory
    filenames = glob.glob(os.path.join(config.img_dir, '*.jpg'))
    filenames = sorted(filenames)
    
    # Clear output directory, TODO
    print("Clean output directory : " + str(config.output_path))
    rmtree(config.output_path, ignore_errors=True)
    os.mkdir(config.output_path)

    model.eval()
    with torch.no_grad():
        for fn in filenames:
            ImageID = fn.split('\\')[-1].split('_')[0]
            output_filename = os.path.join(config.output_path, '{}_mask.png'.format(ImageID))  
            x = Image.open(fn)
            x = transform(x)
            data_shape = x.shape
            x = torch.unsqueeze(x, 0)
            x = x.to(device)
            output = model(x)
            pred = output.max(1, keepdim=True)[1].reshape((-1, data_shape[1], data_shape[2])) # get the index of the max log-probability
            y = torch.zeros((pred.shape[0], 3, pred.shape[1], pred.shape[2]))
            for k, v in MASK.items():
                y[:,0,:,:][pred == k] = v[0]
                y[:,1,:,:][pred == k] = v[1]
                y[:,2,:,:][pred == k] = v[2]

            y = transforms.ToPILImage()(y.squeeze())
            y.save(output_filename)


    # Evaluate prediction results
    if EVAL:
        from mean_iou_evaluate import read_masks, mean_iou_score
        pred = read_masks(config.output_path)
        labels = read_masks('../p2_data/validation/')
        mean_iou_score(pred, labels)

