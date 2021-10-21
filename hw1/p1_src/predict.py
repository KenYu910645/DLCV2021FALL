import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
from vgg16_aug import Vgg16_Aug

INPUT_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',     type=str, default='../p1_data/val_50/')
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--ckpt_path',   type=str,)
    config = parser.parse_args()
    print(config)

    valid_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    # Load model
    vgg16_aug = Vgg16_Aug().to(device)
    state = torch.load(config.ckpt_path)
    vgg16_aug.load_state_dict(state['state_dict'])
    vgg16_aug.eval()

    filenames = glob.glob(os.path.join(config.img_dir, '*.png'))
    filenames = sorted(filenames)

    with open(os.path.join(config.output_path, 'p1_output.csv'), 'w') as f:
        correct = 0  # Debug
        f.write('image_id,label\n')
        with torch.no_grad():
            for fn in filenames:
                x = Image.open(fn)
                x = valid_transform(x)
                x = torch.unsqueeze(x, 0)
                x = x.to(device)
                y = vgg16_aug(x)
                pred = y.max(1, keepdim=True)[1] # get the index of the max log-probability
                f.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')
                
                if pred.item() == int(fn.split('/')[-1].split('_')[0]): # Debug
                    correct += 1
    print("Acurracy = " + str(correct / len(filenames))) # Debug
