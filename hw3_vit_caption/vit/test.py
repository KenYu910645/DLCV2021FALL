#!/usr/bin/env python3

# This sciprt comes from https://github.com/rwightman/pytorch-image-models/blob/master/inference.py

"""PyTorch Inference Script
An example inference script that outputs top-k class ids for images in a folder into a csv.
Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import numpy as np
import torch

from timm.models import create_model
from timm.data import ImageDataset, create_loader
from timm.utils import AverageMeter

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--input_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_path', metavar='DIR',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='vit_base_patch16_224',
                    help='model architecture (default: dpn92)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--topk', default=1, type=int,
                    metavar='N', help='Top-k to output to CSV')

def main():
    args = parser.parse_args()
    # create model
    model = create_model(
        args.model,
        num_classes=37,
        in_chans=3,
        pretrained=False,
        checkpoint_path=args.checkpoint)
    model = model.cuda()

    loader = create_loader(
        ImageDataset(args.input_dir),
        input_size=(3, 224, 224),
        batch_size=32,
        use_prefetcher=True,
        interpolation='bicubic',
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        num_workers=2,
        crop_pct=0.9)

    model.eval()

    k = min(args.topk, 37)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            input = input.cuda()
            labels = model(input)
            topk = labels.topk(k)[1]
            topk_ids.append(topk.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    topk_ids = np.concatenate(topk_ids, axis=0)

    with open(args.output_path, 'w') as f:
        filenames = loader.dataset.filenames(basename=True)
        f.write('filename,label\n')
        if args.eval:
            accuracy = 0 # TODO 
        for filename, label in zip(filenames, topk_ids):
            if args.eval:
                gt_label = os.path.split(filename)[1].split('_')[0] # TODO this should be deleted.
                if int(gt_label) == int(label[0]):
                    accuracy += 1/len(filenames)
            f.write('{0},{1}\n'.format(filename, ','.join([ str(v) for v in label])))

    # TODO 
    if args.eval:
        print(f"accuracy = {accuracy}")

if __name__ == '__main__':
    main()