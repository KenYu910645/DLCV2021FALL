import torch.nn as nn
import torchvision.models as models

class Vgg16_Aug(nn.Module):
    def __init__(self):
        super(Vgg16_Aug, self).__init__()
        self.vgg16 = models.vgg16_bn(pretrained=True)        
        self.vgg16.classifier[6] = nn.Linear(4096, 50)

    def forward(self, x):
        x = self.vgg16(x)
        return x