# coding=utf-8
import torch
import time

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from torch.autograd import Variable
from torch.nn.modules import Module
import numpy as np


class LabelSmoothingLoss(Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x,target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class PrototypicalLoss(Module):
    """
    Loss class deriving from Module for the prototypical loss function defined below
    """

    def __init__(self, n_way, n_support, n_query):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support
        self.n_way = n_way
        self.n_query = n_query
        self.loss_fn = torch.nn.CrossEntropyLoss()

    '''def forward(self, input, target):
        return prototypical_loss(input, target, self.n_way, self.n_support, self.n_query, self.loss_fn)'''

    def forward(self, input):
        return prototypical_loss(input, self.n_way, self.n_support, self.n_query, self.loss_fn)


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, n_way, n_support, n_query, loss_fn):
    """
    """
    z = input.view(n_way, n_support + n_query, -1)
    z_support = z[:, :n_support]
    z_query = z[:, n_support:]

    # contiguous()函数的作用：把tensor变成在内存中连续分布的形式。
    prototypes = z_support.contiguous().view(n_way, n_support, -1).mean(1)
    query_samples = z_query.contiguous().view(n_way * n_query, -1)
    dists = euclidean_dist(query_samples, prototypes)
    # dist (n_query, n_way)
    y_label = np.repeat(range(n_way), n_query)
    # y_label (n_way*n_query, ) e,g,(0,0,0,0,0,1,1,1,1,1......)
    y_query = torch.from_numpy(y_label).long().cuda()
    y_query = Variable(y_query)
    loss_val = loss_fn(-dists, y_query)

    #  k, dim=None, largest=True, sorted=True    return values， indices
    # 找到每一个query样本距离最近的（最接近）的prototype
    topk_scores, topk_labels = (-dists).data.topk(1, 1, True, True)
    # topk_labels (n_query, 1)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = np.sum(topk_ind[:, 0] == y_label)


    acc_val = top1_correct / (topk_ind.shape[0]) * 100

    return loss_val, acc_val


def stl_cls_scores(input, n_way, n_support, n_query, penalty_C):
    z = input.view(n_way, n_support + n_query, -1)
    z_support = z[:, :n_support]
    z_query = z[:, n_support:]

    z_support = z_support.contiguous().view(n_way * n_support, -1)
    z_query = z_query.contiguous().view(n_way * n_query, -1)

    qry_norm = torch.norm(z_query, p=2, dim=1).unsqueeze(1).expand_as(z_query)
    spt_norm = torch.norm(z_support, p=2, dim=1).unsqueeze(1).expand_as(z_support)
    qry_normalized = z_query.div(qry_norm + 1e-6)
    spt_normalized = z_support.div(spt_norm + 1e-6)

    z_query = qry_normalized.detach().cpu().numpy()
    z_support = spt_normalized.detach().cpu().numpy()
    # z_support = np.tile(z_support, (5, 1))

    y_support = np.repeat(range(n_way), n_support)
    # y_support = np.tile(y_support, (5))
    clf = LogisticRegression(penalty='l2',
                             random_state=0,
                             C=penalty_C,
                             solver='lbfgs',
                             max_iter=1000,
                             multi_class='multinomial')
    clf.fit(z_support, y_support)
    y = np.repeat(range(n_way), n_query)
    scores = clf.score(z_query, y)*100
    return scores
