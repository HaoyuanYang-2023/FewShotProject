import os
import torch

import numpy as np

import matplotlib.pyplot as plt

from torchvision import transforms


class PrototypicalBatchSampler(object):
    """
    Adopted from
    https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_batch_sampler.py

    Yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    Each iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').

    ----------------------------------------------------------------------------------------------
    Parameters:
        labels: ndarray, all labels for current dataset
        classes_per_episode: int, number of classes in one episode
        sample_per_class: int, numer of sample in one class
        iterations: int, number of episodes in one epoch
    """

    def __init__(self, labels, classes_per_episode, sample_per_class, iterations, dataset_name="miniImageNet_train"):
        """
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be inferred from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        """
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_episode
        self.sample_per_class = sample_per_class
        self.iterations = iterations
        self.dataset_name = dataset_name

        # 该函数是去除数组中的重复数字，并进行排序之后输出
        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        # indexes
        #                max(count)
        # class:0                        sample_num: 10
        # class:1                        sample_num: 3
        # class:2                        sample_num: 5
        # class:3                        sample_num: 6
        # indexes shape: (4, 40)
        indexes_path = os.path.join(os.getcwd() + '/episode_idx', self.dataset_name + '_indexes.npy')
        numel_per_class_path = os.path.join(os.getcwd() + '/episode_idx', self.dataset_name + '_numel_per_class.npy')
        if not os.path.exists(indexes_path) and not os.path.exists(numel_per_class_path):
            print("Creat dataset indexes")
            self.idxs = range(len(self.labels))
            self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
            self.indexes = torch.Tensor(self.indexes)
            self.numel_per_class = torch.zeros_like(self.classes)
            for idx, label in enumerate(self.labels):
                label_idx = np.argwhere(self.classes == label).item()
                # 即np.where(condition),只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。
                # 这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。
                # 这里会返回第label_idx行中，是nan的坐标, 格式((),)，所以取[0][0]， 获得第一个为nan的坐标
                # 给indexes对应class对应sample赋予idx (在label数组中的idx)
                self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
                self.numel_per_class[label_idx] += 1
            save_path = os.path.join(os.getcwd(), 'episode_idx')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(indexes_path, self.indexes)
            np.save(numel_per_class_path, self.numel_per_class)
        else:
            print("Read Dataset indexes.")
            self.indexes = torch.tensor(np.load(indexes_path))
            self.numel_per_class = torch.tensor(np.load(numel_per_class_path))

    def __iter__(self):
        """
        yield a batch (episode) of indexes
        """
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            # 随机选取c个类
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                # 从第i个class到第i+1个class在batch中的slice
                s = slice(i * spc, (i + 1) * spc)
                # 找到第i个类的label_idx
                label_idx = c.item()
                # 在第label_idx类中随机选择spc个样本
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                # 这些样本的索引写如batch
                batch[s] = self.indexes[label_idx][sample_idxs]
            # 随机打乱batch
            # batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        """
        returns the number of iterations (episodes, batches) per epoch
        """
        return self.iterations


def visual_batch(dataloader, dataset_name):
    """
    Visualize images.
    :param dataloader: dataloader
    :param y: dataset_name, dataset_name
    :return:
    """
    x, y = next(iter(dataloader))
    plt.figure(figsize=(12, 12))
    for i in range(x.shape[0]):
        plt.subplot(5, 6, i + 1)
        idx = y[i].item() - 1
        plt.title(y[i].item())
        plt.imshow(x[i].permute(1, 2, 0))
        plt.axis('off')

    if not os.path.exists(os.path.join(os.getcwd(), 'imgs')):
        os.makedirs(os.path.join(os.getcwd(), 'imgs'))
    plt.savefig('./imgs/visual_batch_' + dataset_name + '.png')


def get_transformers(phase=None):
    if phase not in ["train", "val", "test"]:
        raise ValueError("phase should have value of one of [train, val, test]")
    normalize = transforms.Normalize(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
    if phase == "train":
        transformer = transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transformer = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            normalize,
        ])
    return transformer
