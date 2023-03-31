import torch
import torch.nn as nn

from representation.bdc_module import BDC
from .resnet import ResNet12, resnet18


class META_BDC(nn.Module):
    def __init__(self, model_func, reduced_dim=640):
        super(META_BDC, self).__init__()
        self.model_func = model_func
        # self.feat_dim = int(reduced_dim * (reduced_dim + 1) / 2)
        self.reduced_dim = reduced_dim
        self.dcov = BDC(is_vec=True, input_dim=self.model_func.feat_dim, dimension_reduction=self.reduced_dim)

    def forward(self, x):
        x = self.model_func(x)
        x = self.dcov(x)
        x = x.view(x.size(0), -1)
        return x


def metabdcresnet12(reduce_dim):
    model_func = ResNet12()
    model = META_BDC(model_func=model_func, reduced_dim=reduce_dim)
    return model


def metabdcresnet18(reduce_dim):
    model_func = resnet18()
    model = META_BDC(model_func=model_func, reduced_dim=reduce_dim)
    return model
