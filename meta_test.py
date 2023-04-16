import argparse
import time
import numpy as np
import torch
from torch.cuda.amp import autocast

from tqdm import tqdm
from params import get_params
from Dataset import PrototypicalBatchSampler, get_transformers
from loss import accuracy, stl_cls_scores, euclidean_dist
from utils import model_utils
from utils.model_utils import load_model
from torchvision.datasets import ImageFolder

param = argparse.ArgumentParser()
param.add_argument('--dataset', type=str, default='miniImageNet', help='miniImageNet or tieredImageNet')
param.add_argument('--test_root', type=str, default='./data/miniImageNet/images', help='test root')
param.add_argument('--meta_model_path', type=str, default='./checkpoints/miniImageNet/ResNet10/ResNet10_5way_1shot.pth',
                        help='meta model path')
param.add_argument('--method', type=str, default='protonet', help='protonet or stl')
param.add_argument('--model', type=str, default='ResNet10', help='model name')
param.add_argument('--reduced_dim', type=int, default=64, help='reduced dim')
param.add_argument('--n_way', type=int, default=5, help='n way')
param.add_argument('--n_support', type=int, default=1, help='n support')
param.add_argument('--n_query', type=int, default=15, help='n query')
param.add_argument('--gpu', type=int, default=0, help='gpu id')
param.add_argument('--episodes', type=int, default=2000, help='episodes')
param.add_argument('--pre2meta', type=bool, default=False, help='pre2meta')
param.add_argument('--test_num', type=int, default=3, help='test num')
param.add_argument('--penalty_C', type=float, default=2.0, help='penalty C')
param.add_argument('--num_workers', type=int, default=8, help='num workers')
args = param.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)

if args.method == "stl":
    print("STL Testing......")
else:
    print("Meta Test......")
model = model_utils.model_dict[args.model](args.reduced_dim)
model = load_model(model, args.meta_model_path, pre2meta=args.pre2meta)
model = model.to(device)
model.eval()
model = torch.compile(model, disable=True)

test_trans = get_transformers(phase='test')
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
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=sampler, num_workers=args.num_workers)
acc_list = []
time_list = []

for n in range(args.test_num):
    episode_test_acc = []
    val_time = []

    test_iter = iter(test_dataloader)
    tqdm_gen = tqdm(test_iter)
    for batch in tqdm_gen:
        x, y = batch
        x = x.to(device)
        y_label = np.repeat(range(args.n_way), args.n_query)
        with torch.no_grad():
            with autocast():
                start_time = time.time()
                model_output = model(x)
                if args.method == "stl":
                    acc = stl_cls_scores(model_output, args.n_way, args.n_support, args.n_query, y_label, args.penalty_C)
                if args.method == "protonet":
                    z = model_output.view(args.n_way, args.n_support + args.n_query, -1)
                    z_support = z[:, :args.n_support]
                    z_query = z[:, args.n_support:]
                    prototypes = z_support.contiguous().view(args.n_way, args.n_support, -1).mean(1)
                    query_samples = z_query.contiguous().view(args.n_way * args.n_query, -1)
                    dists = euclidean_dist(query_samples, prototypes)
                    acc = accuracy(dists,y_label)

        end_time = time.time()
        episode_test_acc.append(acc)
        val_time.append(end_time - start_time)
        tqdm_gen.set_description('Avg acc: {:.2f} '
                                 'Avg time: {:.2f}ms '
                                 'curr time: {:.2f}ms'.format(np.mean(episode_test_acc), np.mean(val_time) * 1000,
                                                              (end_time - start_time) * 1000))
    avg_acc = np.mean(episode_test_acc)
    std_acc = np.std(episode_test_acc)
    avg_time = np.mean(val_time) * 1000
    print('Avg Val Acc: {:.2f}%+-{:.2f}%\t'.format(
        avg_acc, 1.96 * std_acc / np.sqrt(args.episodes)))
    print('Inference time for per episode is: {:.2f}ms'.format(avg_time))

    acc_list.append(avg_acc)
    time_list.append(avg_time)

print('Avg acc for {} test round is {:.2f}%'.format(args.test_num, np.mean(acc_list)))
print('Avg time for {} test round is {:.2f}ms'.format(args.test_num, np.mean(time_list)))
