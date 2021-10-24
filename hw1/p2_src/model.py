import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from unet_parts import DoubleConv, Down, Up, OutConv

class FCN32s(nn.Module):
    def __init__(self):
        super(FCN32s, self).__init__()
        self.vgg16_feature = models.vgg16(pretrained=True).features
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
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        bilinear=True
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 7)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class FCN8s(nn.Module):
    # https://zhuanlan.zhihu.com/p/73965733
    def __init__(self, pretrained_net):
        super().__init__()
        self.vgg16_feature = models.vgg16(pretrained=True).features
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 7, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)

if __name__ == '__main__':
    INPUT_SIZE = 512
    print(FCN32s())
    summary(FCN32s().to("cuda"), (3, INPUT_SIZE, INPUT_SIZE))
    # print(UNet())
    # summary(UNet().to("cuda"), (3, INPUT_SIZE, INPUT_SIZE))