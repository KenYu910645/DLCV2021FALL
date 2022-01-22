# import torchvision
# import torch.nn as nn
# import torch

# class Resnet50_ssl(nn.Module):
#     def __init__(self):
#         super(Resnet50_ssl, self).__init__()
#         self.resnet50 = torchvision.models.resnet50(pretrained=False) # TODO fine-tune the MLP layers
#         self.resnet50.fc = nn.Sequential(
#             nn.Linear(2048, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 65)
#         )
#     def forward(self, x):
#         return self.resnet50(x)

# if __name__=="__main__":
#     print(Resnet50_ssl())
#     from torchsummary import summary
#     summary(Resnet50_ssl().to("cuda"), (3, 128, 128))