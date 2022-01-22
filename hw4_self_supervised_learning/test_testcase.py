import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import random
import numpy as np
import pandas as pd
import os
import argparse
from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# added by spiderkiller
from protonet.utils import pairwise_distances
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
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

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        print(episode_file_path)
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def predict(args, model, data_loader):
    for _, m in model.items():
        m.eval()

    prediction_results = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            
            # Put data to gpu 
            data = data.to(DEVICE)

            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:] 
            query_input   = data[args.N_way * args.N_shot:,:,:,:]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            # TODO: extract the feature of support and query data
            support = model['proto'](support_input)
            queries = model['proto'](query_input)

            # TODO: calculate the prototype for each class according to its support data
            prototypes = support.reshape(args.N_way, args.N_shot, -1).mean(dim=1)

            if args.matching_fn == 'parametric':
                distances = pairwise_distances(queries, prototypes, args.matching_fn, model['parametric'])
            else:
                distances = pairwise_distances(queries, prototypes, args.matching_fn)

            # TODO: classify the query data depending on the its distense with each prototype
            y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
            prediction_results.append(y_pred.reshape(-1))

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")
    # Add by spiderkiller
    parser.add_argument('--matching_fn', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # load your model
    state = torch.load(args.load)
    model = {}
    from protonet.model import Protonet
    model['proto'] = Protonet().to(DEVICE)
    model['proto'].load_state_dict(state['state_dict'])

    if args.matching_fn == 'parametric':
        model['parametric'] = nn.Sequential( # TODO 
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(400, 1)
        ).to(DEVICE)
        model['parametric'].load_state_dict(state['parametric'])

    prediction_results = predict(args, model, test_loader)

    # output your prediction to csv
    with open(args.output_csv, 'w') as out_file:
        line = 'episode_id'
        for i in range(args.N_way*args.N_query):
            line += f',query{i}'
        line += '\n'
        out_file.write(line)

        for i, prediction in enumerate(prediction_results):
            line = f'{i}'
            for j in prediction:
                line += f',{j}'
            line += '\n'
            out_file.write(line)