import argparse
import time
import random
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast

from tqdm import tqdm
from params import get_params
from Dataset import PrototypicalBatchSampler, get_transformers
from loss import PrototypicalLoss, stl_cls_scores
from torch.utils.tensorboard import SummaryWriter
from utils import model_utils
from utils.model_utils import load_model
from sklearn.linear_model import LogisticRegression
from torchvision.datasets import ImageFolder

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
"""
Disable cudnn to maximize reproducibility
"""
'''random.seed(args.manual_seed)
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False'''

if args.method == "stl":
    print("Test STL")
else:
    print("Test")
model = model_utils.model_dict[args.model](args.reduced_dim)
model = load_model(model, args.meta_model_path, pre2meta=args.pre2meta)
model = model.to(device)

test_trans = get_transformers('test')
test_dataset = ImageFolder(root=args.test_root, transform=test_trans)
classes_per_it = args.n_way
# 同时产生 support和query
num_samples = args.n_support + args.n_query
sampler = PrototypicalBatchSampler(labels=test_dataset.targets,
                                   classes_per_episode=classes_per_it,
                                   sample_per_class=num_samples,
                                   iterations=args.episodes,
                                   dataset_name='miniImageNet_test' if args.dataset == "miniImageNet" else
                                   'tieredImageNet_test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=sampler, num_workers=0)

episode_test_acc = []
val_time = []
model.eval()
val_iter = iter(test_dataloader)
loss_fn = PrototypicalLoss(n_way=args.n_way, n_support=args.n_support, n_query=args.n_query)

test_iter = iter(test_dataloader)
tqdm_gen = tqdm(test_iter)
for batch in tqdm_gen:
    # for batch in tqdm(val_iter):
    x, y = batch
    x = x.to(device)

    with torch.no_grad():
        with autocast():
            start_time = time.time()
            model_output = model(x)
            if args.method == "stl":
                acc = stl_cls_scores(model_output, args.n_way, args.n_support, args.n_query, args.penalty_C)
            if args.method == "protonet":
                loss, acc = loss_fn(model_output)
            end_time = time.time()
    episode_test_acc.append(acc)
    val_time.append(end_time - start_time)
    tqdm_gen.set_description('Avg acc: {:.2f} '
                             'Avg time: {:.2f}ms '
                             'curr time: {:.2f}ms'.format(np.mean(episode_test_acc),np.mean(val_time) * 1000,(end_time - start_time)*1000))
avg_acc = np.mean(episode_test_acc)
std_acc = np.std(episode_test_acc)
avg_time = np.mean(val_time) * 1000
print('Avg Val Acc: {:.2f}%+-{:.2f}%\t'.format(
    avg_acc, 1.96 * std_acc / np.sqrt(args.episodes)))
print('Inference time for per episode is: {:.2f}ms'.format(avg_time))
