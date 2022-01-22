import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)

# mini-Imagenet dataset, This is from test_testcase.py
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            # Data augumentation
            # transforms.ColorJitter(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=10),
            
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)