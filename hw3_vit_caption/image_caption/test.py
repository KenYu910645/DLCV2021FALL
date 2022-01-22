import torch
from transformers import BertTokenizer
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from models import caption
from datasets import coco
from configuration import Config
import os
import numpy as np
import math 

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

@torch.no_grad()
def evaluate(model, image, caption, cap_mask):
    model.eval()
    for i in range(config.max_position_embeddings - 1):

        predictions, atten_map = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        if predicted_id[0] == 102: # Reach [PAD]
            return caption, atten_map

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption, atten_map

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--input_dir', type=str, help='directory to input image', required=True)
    parser.add_argument('--output_dir', type=str, help='directory to output image', required=True)
    parser.add_argument('--v', type=str, help='version', default='v3')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint

    config = Config()

    if args.v == 'v1':
        model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
    elif args.v == 'v2':
        model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
    elif args.v == 'v3':
        model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    else:
        pass
    
    for image_name in os.listdir(args.input_dir):
        # Load image 
        image = Image.open( os.path.join(args.input_dir, image_name) )
        image = coco.val_transform(image)
        image = image.unsqueeze(0)

        # Token
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
        end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
        
        # Create caption
        caption, cap_mask = create_caption_and_mask(
            start_token, config.max_position_embeddings)
    
        output, atten_map = evaluate(model, image, caption, cap_mask)
        result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        print(result.capitalize())

        n_word = 0
        for j, word in enumerate(caption[0]):
            if word == 0:
                n_word = j+1
                break
        
        img_ori = Image.open( os.path.join(args.input_dir, image_name) ).convert('RGB')
        N_COL = 4
        N_ROW = math.ceil(n_word/N_COL)
        FONT_SIZE = 25
        fig, ax_list = plt.subplots(nrows=N_ROW, ncols=N_COL, figsize=(12,3*N_COL))
        fig.tight_layout()
        [b.axis('off') for a in ax_list for b in a ]# Delete all axis
        
        for j in range(n_word):
            if j == 0:
                ax = ax_list[0][0]
                ax.set_title("<start>", fontsize=FONT_SIZE)
                ax.imshow(img_ori, alpha=1)
                continue
            # Denormalize
            single_atten_map = atten_map[0][j-1]
            single_atten_map = single_atten_map*(255.0 / torch.max(single_atten_map))
            single_atten_map = single_atten_map.reshape((math.floor(single_atten_map.shape[0]/19), 19))
            atten_img = Image.fromarray(np.array(single_atten_map, dtype=np.uint8))
            
            # Get token 
            token = tokenizer.decode(caption[0][j].tolist(), skip_special_tokens=False).replace(" ", "")
            
            # Get subplot
            ax = ax_list[math.floor(j/N_COL)][math.floor(j%N_COL)]
            if token == "[CLS]":
                ax.set_title("<start>", fontsize=FONT_SIZE)
            elif token == "[PAD]":
                ax.set_title("<end>", fontsize=FONT_SIZE)
            else:
                ax.set_title(token, fontsize=FONT_SIZE)
            ax.imshow(img_ori, alpha=1)
            atten_img = atten_img.resize(img_ori.size)
            ax.imshow(atten_img, alpha=0.5, interpolation='nearest', cmap="jet")
        fig.savefig(os.path.join(args.output_dir, os.path.splitext(image_name)[0] + '.png'))
