from torch import nn

# Conv-4 Network, This is provided by TA
class Convnet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Protonet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.conv = Convnet(in_channels, hid_channels, out_channels)

        # This is not used 
        self.mlp = nn.Sequential(
            nn.Linear(1600, 800),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(800, 800),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(800, 400)
        )

    def forward(self, x):
        x = self.conv(x)
        # x = self.mlp(x)
        # return x

        # reference https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/protonet.py
        return x.view(x.size(0), -1)


class Parametric_distance(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3200, 1600),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1600, 800),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(400, 1))
    def forward(self, x):
        return self.layer(x)

if __name__ == "__main__":
    # print(Convnet())
    # print(Protonet())
    print(Parametric_distance())

