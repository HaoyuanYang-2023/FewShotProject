import argparse
import json
import os
import time
import random
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from params import get_params
from Dataset import PrototypicalBatchSampler, get_transformers
from loss import PrototypicalLoss
from torch.utils.tensorboard import SummaryWriter
from utils import model_utils,init_lr_scheduler,init_seed
from utils.model_utils import load_model
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser()
parser.add_argument('--param_file', type=str, default=None, help="JSON file for parameters")

parser.add_argument('--dataset', type=str, help='dataset name', default='miniImageNet',
                        choices=['miniImageNet', 'tieredImageNet'])
parser.add_argument('--train_root', type=str, help='path to dataset', default='')
parser.add_argument('--val_root', type=str, help='path to dataset', default='')
parser.add_argument('--num_workers', type=int, default=8)

parser.add_argument('--model', type=str, help='model to use', default="MPNCOVResNet12")
parser.add_argument('--pretrain_model_path',type=str,default='')
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--reduced_dim', type=int, default=640, help="Dimensions to reduce before cov layer")

parser.add_argument('--epochs', type=int, default=160)
parser.add_argument('--train_n_episodes', type=int, help='number of val episodes, default=1000', default=1000)
parser.add_argument('--val_n_episodes', type=int, help='number of val episodes, default=600', default=600)
parser.add_argument('--n_way', type=int, default=5)
parser.add_argument('--n_support', type=int, default=5)
parser.add_argument('--n_query', type=int, default=15)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--optimizer', type=str, default='SGD', choices=['Adam', 'SGD'], help="Optimizers")
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--nesterov', type=bool, default=True, help='nesterov momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
parser.add_argument('--milestones', default=[80, 120, 140])

parser.add_argument('--lrG', type=float, help='StepLR learning rate scheduler gamma, default=0.1', default=0.1)
parser.add_argument('--exp', type=str, help='exp information', default='')
parser.add_argument('--print_freq', type=int, help="Step interval to print", default=100)
parser.add_argument('--manual_seed', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--resume', action='store_true', help='resume training')
parser.add_argument('--checkpoint_path', type=str, default='')

args = parser.parse_args()

if args.param_file is not None:
    args = get_params(args.param_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)

print()
print(args)


def train(tr_dataloader, model, optim, lr_scheduler, checkpoint_dir, val_dataloader):

    start_epoch = 0

    best_acc = 0

    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    last_model_path = os.path.join(checkpoint_dir, 'last_model.pth')
    loss_fn = PrototypicalLoss(args.n_way, args.n_support, args.n_query)
    scaler = GradScaler()

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
        episode_tr_loss = []
        episode_tr_acc = []

        print('\n=== Epoch: {} ==='.format(epoch))
        model.train()

        for batch_idx, batch in enumerate(tr_dataloader):
            optim.zero_grad()
            x, y = batch
            x = x.to(device)
            with autocast():
                model_output = model(x)
                loss, acc = loss_fn(model_output)
            episode_tr_loss.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            episode_tr_acc.append(acc)

            if (batch_idx + 1) % 100 == 0:
                print('Iters: {}/{}\t'
                      'Loss: {:.4f}\t'
                      'Prec@1 {:.2f}\t'.format(batch_idx + 1, args.train_n_episode, np.mean(episode_tr_loss),
                                               np.mean(episode_tr_acc)))
        avg_loss = np.mean(episode_tr_loss)
        avg_acc = np.mean(episode_tr_acc)

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

            with autocast():
                with torch.no_grad():
                    start_time = time.time()
                    model_output = model(x)
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
            print("best model saved!")

    torch.save(model.state_dict(), last_model_path)


def init_dataset():
    tr_trans = get_transformers('train')
    val_trans = get_transformers('val')
    train_dataset = ImageFolder(root=args.train_root, transform=tr_trans)
    val_dataset = ImageFolder(root=args.val_root,
                              transform=val_trans)
    n_classes = len(np.unique(train_dataset.targets))
    if n_classes < args.n_way:
        raise (Exception('There are not enough classes in the dataset in order ' +
                         'to satisfy the chosen n_way. Decrease the ' +
                         'n_way option and try again.'))
    return train_dataset, val_dataset


def init_sampler(labels, mode):
    classes_per_it = args.n_way
    # 同时产生 support和query
    num_samples = args.n_support + args.n_query
    if 'train' == mode:
        sampler = PrototypicalBatchSampler(labels=labels,
                                           classes_per_episode=classes_per_it,
                                           sample_per_class=num_samples,
                                           iterations=args.train_n_episode,
                                           dataset_name=args.dataset + '_train')
    else:
        sampler = PrototypicalBatchSampler(labels=labels,
                                           classes_per_episode=classes_per_it,
                                           sample_per_class=num_samples,
                                           iterations=args.val_n_episode,
                                           dataset_name=args.dataset + '_test')
    return sampler


def init_dataloader():
    train_dataset, val_dataset = init_dataset()
    tr_sampler = init_sampler(train_dataset.targets, 'train')
    val_sampler = init_sampler(val_dataset.targets, 'val')
    tr_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=tr_sampler, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=args.num_workers)
    return tr_dataloader, val_dataloader


def init_model():
    model = model_utils.model_dict[args.model](args.reduced_dim).to(device)

    model = load_model(model, args.pretrain_model_path)
    return model


def init_optim(model):
    """
    Initialize optimizer
    """
    if args.optimizer == "Adam":
        return torch.optim.Adam(params=model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    if args.optimizer == "SGD":
        return torch.optim.SGD(params=model.parameters(),
                               nesterov=args.nesterov,
                               lr=args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)


init_seed(args.manual_seed)
tr_dataloader, val_dataloader = init_dataloader()

model = init_model()
optim = init_optim(model)
lr_scheduler = init_lr_scheduler(optim,args.lrG,args.milestones)

if __name__ == "__main__":
    checkpoint_dir = 'runs/%s/%s' % (args.dataset, args.model)
    checkpoint_dir += '_%dway_%dshot' % (args.n_way, args.n_support)
    checkpoint_dir += '_metatrain_' + args.exp
    if not os.path.exists(os.path.join(os.getcwd(), checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(), checkpoint_dir))
    print(checkpoint_dir)

    with open(checkpoint_dir + '/params.json', mode="wt") as f:
        json.dump(vars(args), f, indent=4)

    logger = SummaryWriter(log_dir=checkpoint_dir)

    train(tr_dataloader=tr_dataloader, val_dataloader=val_dataloader, model=model, lr_scheduler=lr_scheduler,
          optim=optim, checkpoint_dir=checkpoint_dir)
