import argparse
import json
import os
import time
import random

import torch
import torch.nn as nn
import numpy as np
from params import get_params
from Dataset import get_transformers
from model.mpncovresnet import mpncovresnet12, A_MPNCOV
from torchvision.datasets import ImageFolder

from representation.MPNCOV import MPNCOV

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--train_root', default='F:/miniImageNet/train', type=str)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print()
print(args)


def get_memory_total():
    last_memory = torch.cuda.memory_allocated() / 1024 / 1024
    return last_memory


def accuracy(model_output, y):
    _, pred = model_output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y.long().view(1, -1).expand_as(pred))
    acc = correct.sum().item() / y.size(0)
    return acc


def grad_test(model):
    back_time = []
    back_memory = []
    model.train()

    for batch_idx in range(100):
        x = torch.randn([1, 640, 10, 10], requires_grad=True).to(device)
        y = torch.ones([1, 1]).long().to(device)
        model_output = model(x)
        loss = torch.nn.CrossEntropyLoss()(model_output, y)
        t = time.time()
        loss.backward()
        t2 = time.time()
        m = get_memory_total()

        back_memory.append(m)
        back_time.append((t2 - t) * 1000)
        if batch_idx + 1 == 100:
            break
        print('Iters: [{}/{}], '
              'Avg Back Time: {:.2f}ms, '
              'curr time: {:.2f}ms, '
              'Avg Back Memory: {:.2f}MB '
              'curr Memory: {:.2f}MB'.format(batch_idx + 1, len(tr_dataloader),
                                             np.mean(back_time),
                                             (t2 - t) * 1000,
                                             np.mean(back_memory),
                                             m))



def init_seed():
    """
    Disable cudnn to maximize reproducibility
    """
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_dataset():
    tr_trans = get_transformers('train')
    train_dataset = ImageFolder(root=args.train_root,
                                transform=tr_trans)
    return train_dataset


train_dataset = init_dataset()

tr_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=0)

init_seed()
model = A_MPNCOV(iterNum=3).to(device)
model2 = MPNCOV(iterNum=3).to(device)
if __name__ == "__main__":
    grad_test(model=model)
