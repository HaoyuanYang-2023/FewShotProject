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
param.add_argument('--param_file', type=str, default=None, help="JSON file for parameters")
json_parm = param.parse_args()
print(json_parm)
args = get_params(json_parm.param_file)

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
