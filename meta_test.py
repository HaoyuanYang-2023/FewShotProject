import argparse
import time

import numpy as np
import torch
from torch.cuda.amp import autocast

from tqdm import tqdm
from params import get_params
from Dataset import ImageNetFolder, PrototypicalBatchSampler, get_transformers
from loss import PrototypicalLoss
from torch.utils.tensorboard import SummaryWriter
from utils import model_utils

param = argparse.ArgumentParser()
param.add_argument('--param_file', type=str, default=None, help="JSON file for parameters")
json_parm = param.parse_args()
print(json_parm)
if json_parm.param_file is not None:
    args = get_params(True, json_parm.param_file)
else:
    args = get_params()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)

model = model_utils.model_dict[args.model]().to(device)

model.load_state_dict(torch.load(args.meta_model_path))
if hasattr(model, 'reduced_dim') and args.reduced_dim is not None:
    model.reduced_dim = args.reduced_dim

test_trans = get_transformers('test')
test_dataset = ImageNetFolder(root=args.test_root, dataset_name=args.dataset, phase='test',
                              transformer=test_trans)
classes_per_it = args.n_way
# 同时产生 support和query
num_samples = args.n_support + args.n_query
sampler = PrototypicalBatchSampler(labels=test_dataset.targets,
                                   classes_per_episode=classes_per_it,
                                   sample_per_class=num_samples,
                                   iterations=args.episodes,
                                   dataset_name='miniImageNet_test' if args.dataset == "miniImageNet" else
                                   'tieredImageNet_test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=sampler)
episode_test_loss = []
episode_test_acc = []
val_time = []
model.eval()
val_iter = iter(test_dataloader)
loss_fn = PrototypicalLoss(n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)

test_iter = iter(test_dataloader)
for batch in tqdm(test_iter):
    # for batch in tqdm(val_iter):
    x, y = batch
    x = x.to(device)
    start_time = time.time()
    with autocast():
        model_output = model(x)
        loss, acc = loss_fn(model_output)
    end_time = time.time()
    episode_test_loss.append(loss.item())
    episode_test_acc.append(acc)
    val_time.append(end_time - start_time)
avg_loss = np.mean(episode_test_loss)
avg_acc = np.mean(episode_test_acc)
std_acc = np.std(episode_test_acc)
avg_time = np.mean(val_time) * 1000
print('Avg Val Loss: {:.4f}, Avg Val Acc: {:.2f}%+-{:.2f}%\t'.format(
            avg_loss, avg_acc, 1.96 * std_acc / np.sqrt(args.episodes)))
print('Inference time for per episode is: {:.2f}ms'.format(avg_time))