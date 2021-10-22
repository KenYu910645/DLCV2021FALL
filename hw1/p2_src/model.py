import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

class FCN32s(nn.Module):
    def __init__(self):
        super(FCN32s, self).__init__()
        self.vgg16_feature = models.vgg16(pretrained=True).features
        # self.conv6 = nn.Conv2d(512, 4096, 1)
        # self.relu6 = nn.ReLU(inplace=True)
        # self.drop6 = nn.Dropout2d()
        # self.conv7 = nn.Conv2d(4096, 4096, 1)
        # self.relu7 = nn.ReLU(inplace=True)
        # self.drop7 = nn.Dropout2d()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 7, 1)
        )
        self.conv_up32 = nn.ConvTranspose2d(7, 7, 64, stride=32, bias=False, padding=16)

    def forward(self, x):
        x = self.vgg16_feature(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv_up32(x)

        # x_shape = x.shape
        # x = self.drop6(self.relu6(self.conv6(x)))
        # x = self.drop7(self.relu7(self.conv7(x)))
        # x = self.score(x)
        # x = self.upsample32(x)
        # x = x[:, :, 16: 16 + x_shape[2], 16: 16 + x_shape[3]]
        return x

if __name__ == '__main__':
    INPUT_SIZE = 512
    print(FCN32s())
    summary(FCN32s().to("cuda"), (3, INPUT_SIZE, INPUT_SIZE))