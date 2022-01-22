import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from model import Protonet, Parametric_distance
from utils import pairwise_distances
import random
from dataset import MiniDataset
from samplers import GeneratorSampler, NShotTaskSampler
from torch.utils.data import DataLoader

DEVICE = 'cuda'
TRAIN_CSV = "../hw4_data/mini/train.csv"
TRAIN_IMG = "../hw4_data/mini/train/"
VAL_CSV = "../hw4_data/mini/val.csv"
VAL_IMG = "../hw4_data/mini/val/"
VAL_TEST_CSV = "../hw4_data/mini/val_testcase.csv"
CKPT_PATH = "../p1_ckpt_k5/"

DISTANCE_FUNCTION = 'l2' # 'parametric' # 'l2' # 'cosine' 'parametric'
N_WAY = 5
N_SHOT =  1
N_WAY_VAL = 5
N_SHOT_VAL = 1
ENABLE_SAVE = True 

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

train_dataset = MiniDataset(TRAIN_CSV, TRAIN_IMG)
val_dataset   = MiniDataset(VAL_CSV, VAL_IMG)

def worker_init_fn(worker_id): # Avoid workers using same seed.                             
    np.random.seed(np.random.get_state()[1][0] + worker_id)

train_loader = DataLoader(
    train_dataset,
    num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
    batch_sampler=NShotTaskSampler(TRAIN_CSV, 600, N_WAY, N_SHOT, 15)) # 600 episode per epoch, 5 way, 1 shot, N_query_train

val_loader = DataLoader(
    val_dataset, batch_size=N_WAY_VAL*(15 + N_SHOT_VAL),
    num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
    sampler=GeneratorSampler(VAL_TEST_CSV))

def train(model, matching_fn, n_epoch = 100, log_interval = 300, ckpt_interval = 600):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,  betas=[0.9, 0.999], weight_decay=1e-2)
    if matching_fn == 'parametric':
        parametric = Parametric_distance().to(DEVICE)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(parametric.parameters()), lr=1e-4,  betas=[0.9, 0.999], weight_decay=1e-2)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.9)

    best_mean = 0
    iteration = 0

    episodic_acc = []
    for ep in range(n_epoch):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            support_input = data[:N_WAY * N_SHOT,:,:,:] 
            query_input   = data[N_WAY * N_SHOT:,:,:,:]

            label_encoder = {target[i * N_SHOT] : i for i in range(N_WAY)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[N_WAY * N_SHOT:]])

            support = model(support_input)
            queries = model(query_input)
            prototypes = support.reshape(N_WAY, N_SHOT, -1).mean(dim=1)

            if matching_fn == 'parametric':
                distances = pairwise_distances(queries, prototypes, matching_fn, parametric)
            else:
                distances = pairwise_distances(queries, prototypes, matching_fn)

            loss = criterion(-distances, query_label)
            loss.backward()
            optimizer.step()

            y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
            episodic_acc.append(1. * y_pred.eq(query_label.view_as(y_pred)).sum().item() / len(query_label))

            if (iteration + 1) % log_interval == 0:
                episodic_acc = np.array(episodic_acc)
                mean = episodic_acc.mean()
                std = episodic_acc.std()

                print(f'Epoch: {ep} [{(batch_idx + 1)}/{len(train_loader)}]  Iteration: {iteration + 1}  Loss: {round(loss.item(), 3)}  Accuracy Mean: {round(mean*100, 2)}% +- { round(1.96*std/(log_interval)**(1/2)*100, 2)} ')
                episodic_acc = []

            if (iteration + 1) % ckpt_interval == 0:
                if matching_fn == 'parametric':
                    loss, mean, std = eval(model, matching_fn, parametric)
                else:
                    loss, mean, std = eval(model, matching_fn)
                if mean > best_mean:
                    best_mean = mean
                if ENABLE_SAVE:
                    # Save checkpoint to directory
                    state = {'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict()}
                    if matching_fn == 'parametric':
                        state = {'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'parametric': parametric.state_dict()}

                    new_checkpoint_path = os.path.join(CKPT_PATH, '{}-protonet.pth'.format(iteration + 1))
                    torch.save(state, new_checkpoint_path)
                    print('model saved to %s' % new_checkpoint_path)

            iteration += 1

        scheduler.step()

def eval(model, matching_fn, parametric = None):
    # This is from test_testcase.py
    criterion = nn.CrossEntropyLoss()
    model.eval()
    episodic_acc = []
    loss = []
    
    with torch.no_grad():
        for b_idx, (data, target) in enumerate(val_loader):
            data = data.to(DEVICE)
            support_input = data[:N_WAY_VAL*N_SHOT_VAL,:,:,:] 
            query_input = data[N_WAY_VAL*N_SHOT_VAL:,:,:,:]

            label_encoder = {target[i*N_SHOT_VAL] : i for i in range(N_WAY_VAL)}

            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[N_WAY_VAL*N_SHOT_VAL:]])

            # extract the feature of support and query data
            support = model(support_input)
            queries = model(query_input)

            # calculate the prototype for each class according to its support data
            prototypes = support.reshape(N_WAY_VAL, N_SHOT_VAL, -1).mean(dim=1)

            if matching_fn == 'parametric':
                distances = pairwise_distances(queries, prototypes, matching_fn, parametric)
            else:
                distances = pairwise_distances(queries, prototypes, matching_fn)
            
            # classify the query data depending on the its distense with each prototype
            loss.append(criterion(-distances, query_label).item())
            y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
            episodic_acc.append(1. * y_pred.eq(query_label.view_as(y_pred)).sum().item() / len(query_label))

    loss = np.array(loss)
    episodic_acc = np.array(episodic_acc)
    loss = loss.mean()
    mean = episodic_acc.mean()
    std = episodic_acc.std()

    print(f'\nLoss: {round(loss, 2)} Accuracy Mean: {round(mean*100, 2)}% +- { round(1.96*std/(600)**(1/2)*100, 2)}\n')

    return loss, mean, std

if __name__ == '__main__':
    model = Protonet().to(DEVICE)
    train(model, DISTANCE_FUNCTION, n_epoch = 30, log_interval = 300, ckpt_interval = 600)