import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from params import get_params
from Dataset import PrototypicalBatchSampler, get_transformers
from loss import PrototypicalLoss
from torch.utils.tensorboard import SummaryWriter
from utils import model_utils
from utils.model_utils import load_model
from torchvision.datasets import ImageFolder

param = argparse.ArgumentParser()
param.add_argument('--param_file', type=str, default=None, help="JSON file for hyper-parameters")
json_parm = param.parse_args()
print(json_parm)
if json_parm.param_file is not None:
    args = get_params(True, json_parm.param_file)
else:
    args = get_params()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)

print()
print(args)


class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss


def train(tr_dataloader, model, model_t, optim, lr_scheduler, checkpoint_dir, val_dataloader=None):
    """
    Train the model with the prototypical learning algorithm
    """

    start_epoch = 0

    best_acc = 0

    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    last_model_path = os.path.join(checkpoint_dir, 'last_model.pth')

    loss_fn = PrototypicalLoss(args.n_way, args.n_support, args.n_query)
    scaler = GradScaler()
    loss_ce = nn.CrossEntropyLoss()
    loss_div_fn = DistillKL(4)
    if args.resume:
        checkpoint = torch.load(args.checkpoint_path)
        assert ValueError("No checkpoint found!")
        model.load_state_dict(checkpoint['state'])
        optim.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])

    for epoch in range(start_epoch, args.epochs):
        tr_loss = []
        tr_acc = []

        print('\n=== Epoch: {} ==='.format(epoch))
        model.train()
        train_iter = iter(tr_dataloader)

        for batch_idx, batch in enumerate(tr_dataloader):
            # for batch in tqdm(train_iter):
            optim.zero_grad()
            x, y = batch
            # print(x.shape)
            # x, y = x.to(device), y.to(device)
            x = Variable(x).to(device)
            y = Variable(y).to(device)

            with autocast():
                with torch.no_grad():
                    model_t_output = model_t(x)
                model_output = model(x)
                loss_cls = loss_ce(model_output, y)

                loss_div = loss_div_fn(model_output, model_t_output)
                loss = loss_cls * 0.5 + loss_div * 0.5
            tr_loss.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optim)  # .step()
            scaler.update()
            # loss.backward()
            # optim.step()
            _, pred = model_output.topk(1, 1, True, True)
            pred = pred.t()
            correct = pred.eq(y.long().view(1, -1).expand_as(pred))
            acc = correct.sum().item() / y.size(0)
            tr_acc.append(acc * 100)
            if (batch_idx + 1) % 100 == 0:
                print('Iters: {}/{}\t'
                      'Loss: {:.4f}\t'
                      'Prec@1 {:.2f}\t'.format(batch_idx + 1, len(tr_dataloader), np.mean(tr_loss),
                                               np.mean(tr_acc)))
        avg_loss = np.mean(tr_loss)
        avg_acc = np.mean(tr_acc)

        print('Avg Train Loss: {:.4f}, Avg Train Acc: {:.2f}%'.format(avg_loss, avg_acc))
        logger.add_scalar("Loss/Train", avg_loss, global_step=epoch)
        logger.add_scalar("Accuracy/Train", avg_acc, global_step=epoch)

        if (epoch + 1) % 50 == 0:
            state = {'state': model.state_dict(),
                     'optimizer': optim.state_dict(),
                     'epoch': epoch + 1,
                     'best_acc': best_acc,
                     'lr_scheduler': lr_scheduler.state_dict(),
                     'scaler': scaler.state_dict()}
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint' + str(epoch + 1) + '.pth')
            torch.save(state, checkpoint_path)
            print("Checkpoint Saved!")
        lr_scheduler.step()
        if val_dataloader is None:
            continue

        episode_val_loss = []
        episode_val_acc = []
        val_time = []
        model.eval()
        for batch_idx, batch in enumerate(val_dataloader):
            x, y = batch
            x = x.to(device)
            start_time = time.time()
            with autocast():
                model_output = model[0](x)
                model_output = model[1](model_output)
                loss, acc = loss_fn(model_output)
            end_time = time.time()
            episode_val_loss.append(loss.item())
            episode_val_acc.append(acc)
            val_time.append(end_time - start_time)
        avg_loss = np.mean(episode_val_loss)
        avg_acc = np.mean(episode_val_acc)
        std_acc = np.std(episode_val_acc)
        avg_time = np.mean(val_time) * 1000
        logger.add_scalar("Loss/Validation", avg_loss, global_step=epoch)
        logger.add_scalar("Accuracy/Validation", avg_acc, global_step=epoch)

        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {:.2f}%)'.format(
            best_acc)
        print('Avg Val Loss: {:.4f}, Avg Val Acc: {:.2f}%+-{:.2f}%\t{}'.format(
            avg_loss, avg_acc, 1.96 * std_acc / np.sqrt(args.val_n_episode), postfix))
        print('Inference time for per episode is: {:.2f}ms'.format(avg_time))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc

    torch.save(model.state_dict(), last_model_path)


def init_seed():
    """
    Disable cudnn to maximize reproducibility
    """
    torch.cuda.cudnn_enabled = False
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)


def init_dataset():
    tr_trans = get_transformers('train')
    val_trans = get_transformers('val')
    train_dataset = ImageFolder(root=args.train_root,
                                transform=tr_trans)
    val_dataset = ImageFolder(root=args.val_root, transform=val_trans)
    n_classes = len(np.unique(train_dataset.targets))
    if n_classes < args.n_way:
        raise (Exception('There are not enough classes in the dataset in order ' +
                         'to satisfy the chosen n_way. Decrease the ' +
                         'n_way option and try again.'))
    return train_dataset, val_dataset


def init_val_sampler(labels):
    classes_per_it = args.n_way
    # 同时产生 support和query
    num_samples = args.n_support + args.n_query
    sampler = PrototypicalBatchSampler(labels=labels,
                                       classes_per_episode=classes_per_it,
                                       sample_per_class=num_samples,
                                       iterations=args.val_n_episode,
                                       dataset_name=args.dataset + '_test')
    return sampler


def init_dataloader():
    train_dataset, val_dataset = init_dataset()
    val_sampler = init_val_sampler(val_dataset.targets)
    tr_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True,
                                                num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=args.num_workers)
    return tr_dataloader, val_dataloader


def init_model():
    """
    Initialize the ProtoNet
    """
    model = model_utils.model_dict[args.model](args.keep_prob, args.reduced_dim)
    in_dim = int(args.reduced_dim * (args.reduced_dim + 1) / 2)
    model_s = nn.Sequential(model, nn.Dropout(args.dropout_rate), nn.Linear(in_dim, args.num_class))
    model_t = nn.Sequential(model, nn.Dropout(args.dropout_rate), nn.Linear(in_dim, args.num_class))
    model_t = model_utils.load_model(model_t, args.pretrain_model_path, pre2meta=False)
    model_s = model_s.to(device)
    model_t = model_t.to(device)
    return model_s, model_t


def init_optim(model):
    """
    Initialize optimizer
    """
    if args.optimizer == "Adam":
        return torch.optim.Adam(params=model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    if args.optimizer == "SGD":
        if "BDC" in args.model:
            bas_params = filter(lambda p: id(p) != id(model[0].dcov.temperature), model.parameters())
            optimizer = torch.optim.SGD([
                {'params': bas_params},
                {'params': model[0].dcov.temperature, 'lr': args.t_lr}], lr=args.lr, weight_decay=args.weight_decay,
                nesterov=True, momentum=0.9)
            return optimizer
        else:
            return torch.optim.SGD(params=model.parameters(),
                                   nesterov=args.nesterov,
                                   lr=args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)


def init_lr_scheduler(optim):
    """
    Initialize the learning rate scheduler
    """
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optim,
                                                gamma=args.lrG,
                                                milestones=args.milestones)


init_seed()
tr_dataloader, val_dataloader = init_dataloader()

model, model_t = init_model()
optim = init_optim(model)
lr_scheduler = init_lr_scheduler(optim)

if __name__ == "__main__":
    checkpoint_dir = 'runs/%s/%s' % (args.dataset, args.model)
    checkpoint_dir += '_distillation_born{}'.format(args.born)
    checkpoint_dir += '_'
    checkpoint_dir += args.exp
    if not os.path.exists(os.path.join(os.getcwd(), checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(), checkpoint_dir))
    print(checkpoint_dir)

    with open(checkpoint_dir + '/params.json', mode="wt") as f:
        json.dump(vars(args), f, indent=4)

    logger = SummaryWriter(log_dir=checkpoint_dir)

    train(tr_dataloader=tr_dataloader, val_dataloader=val_dataloader, model=model, lr_scheduler=lr_scheduler,
          optim=optim, checkpoint_dir=checkpoint_dir, model_t=model_t)
