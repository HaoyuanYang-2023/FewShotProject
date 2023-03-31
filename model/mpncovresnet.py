import torch
import torch.nn as nn
from split_layer import split_layer_dec
from torch.autograd import Function
from model.resnet import ResNet12, ResNet34s, resnet18
from representation import MPNCOV


class A_MPNCOV(nn.Module):
    def __init__(self, iterNum):
        super(A_MPNCOV, self).__init__()
        self.iterN = iterNum
        self.meta_layer1 = covpool()
        self.meta_layer2 = isqrtm(n=3)
        self.meta_layer3 = tritovec()

    def forward(self, x):
        x = self.meta_layer1(x)
        x = self.meta_layer2(x)
        x = self.meta_layer3(x)
        return x


class covpool(nn.Module):
    def __init__(self):
        super(covpool, self).__init__()

    def forward(self, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h * w
        x = x.reshape(batchSize, dim, M)
        I_hat = (-1. / M / M) * torch.ones(M, M, device=x.device) + (1. / M) * torch.eye(M, M, device=x.device)
        I_hat = I_hat.view(1, M, M).repeat(batchSize, 1, 1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1, 2))

        return y


class isqrtm(nn.Module):
    def __init__(self, n):
        super(isqrtm, self).__init__()
        self.iterN = n

    def forward(self, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))

        if self.iterN < 2:
            ZY = 0.5 * (I3 - A)
            YZY = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y = A.bmm(ZY)
            Z = ZY
            for i in range(1, self.iterN - 1):
                ZY = 0.5 * (I3 - Z.bmm(Y))
                Y = Y.bmm(ZY)
                Z = ZY.bmm(Z)
            YZY = 0.5 * Y.bmm(I3 - Z.bmm(Y))
        y = YZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        return y


class tritovec(nn.Module):
    def __init__(self):
        super(tritovec, self).__init__()

    def forward(self, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        # dtype = x.dtype
        x = x.reshape(batchSize, dim * dim)
        I = torch.ones(dim, dim).triu().reshape(dim * dim)
        index = I.nonzero()
        # y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(dtype)
        y = x[:, index]
        return y


class MPNCOVResNet(nn.Module):
    def __init__(self, model_func, reduced_dim=640):
        super(MPNCOVResNet, self).__init__()
        self.model_func = model_func
        self.feat_dim = self.model_func.feat_dim[0]
        self.reduced_dim = reduced_dim
        if self.feat_dim != self.reduced_dim:
            self.reduced_dim = reduced_dim
            self.layer_reduce = nn.Conv2d(self.feat_dim, self.reduced_dim, kernel_size=1, stride=1, padding=0,
                                          bias=False)
            self.layer_reduce_bn = nn.BatchNorm2d(self.reduced_dim)
            self.layer_reduce_relu = nn.ReLU(inplace=True)

        self.meta_layer = MPNCOV.MPNCOV(iterNum=3, input_dim=640, dimension_reduction=reduced_dim)

    def forward(self, x):
        x = self.model_func(x)
        x = self.meta_layer(x)
        x = x.view(x.size(0), -1)
        return x


def mpncovresnet12(reduce_dim):
    model_func = ResNet12()

    model = MPNCOVResNet(model_func=model_func, reduced_dim=reduce_dim)
    return model


def mpncovresnet18(reduce_dim):
    model_func = resnet18()

    model = MPNCOVResNet(model_func=model_func, reduced_dim=reduce_dim)
    return model


def mpncovresnet34(reduce_dim):
    model_func = ResNet34s()
    model = MPNCOVResNet(model_func, reduce_dim)
    return model
