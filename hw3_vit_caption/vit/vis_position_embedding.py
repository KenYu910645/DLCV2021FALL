import os
import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from timm import create_model

model_name = "vit_base_patch16_224"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)
model = create_model(model_name, pretrained=True).to(device)

# Define transforms for test
IMG_SIZE = (224, 224)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
transforms = [
              T.Resize(IMG_SIZE),
              T.ToTensor(),
              T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
              ]
transforms = T.Compose(transforms)

# ImageNet Labels
imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))

# Demo Image
img = PIL.Image.open('/home/lab530/KenYu/hw3-KenYu910645/hw3_data/p1_data_imagenet_format/train/0/0_1847.jpg')
plt.savefig("2.jpg")

img_tensor = transforms(img).unsqueeze(0).to(device)

# end-to-end inference
output = model(img_tensor)

print("Inference Result:")
print(imagenet_labels[int(torch.argmax(output))])
img.save('original.jpg')

#  Split Image into Patches
patches = model.patch_embed(img_tensor)  # patch embedding convolution
print("Image tensor: ", img_tensor.shape)
print("Patch embeddings: ", patches.shape)

#　Add Position Embeddings
pos_embed = model.pos_embed
print(pos_embed.shape)

# Visualize position embedding similarities.
# One cell shows cos similarity between an embedding and all the other embeddings.
# cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Visualization of position embedding similarities", fontsize=24)
for i in range(1, pos_embed.shape[1]): # 197
    sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
    sim = sim.reshape((14, 14)).detach().cpu().numpy()
    ax = fig.add_subplot(14, 14, i)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(sim)

    if i == pos_embed.shape[1]-1:
        fig.savefig("vis_result.jpg")
    