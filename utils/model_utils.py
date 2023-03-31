import os

from model import resnet, meta_bdc
from model import mpncovresnet
import torch

model_dict = dict(
    # ResNet10=resnet.resnet10,
    ResNet12=resnet.ResNet12,
    ResNet18=resnet.resnet18,
    ResNet34=resnet.resnet34,
    # ResNet34s=resnet.resNet34s,
    ResNet50=resnet.resnet50,
    ResNet101=resnet.resnet101,
    ResNet152=resnet.resnet152,
    MPNCOVResNet12=mpncovresnet.mpncovresnet12,
    MPNCOVResNet18=mpncovresnet.mpncovresnet18,
    MPNCOVResNet34=mpncovresnet.mpncovresnet34,
    BDCResNet12=meta_bdc.metabdcresnet12,
    BDCResNet18=meta_bdc.metabdcresnet18)


def load_model(model, dir, pre2meta=True):
    model_dict = model.state_dict()
    file_dict = torch.load(dir)
    if 'state' in file_dict.keys():
        file_dict = torch.load(dir)['state']
    else:
        file_dict = torch.load(dir)
    d = {}
    miss = 0
    # print(model_dict.keys())
    if pre2meta:
        for k, v in file_dict.items():
            if k[2:] in model_dict.keys():
                # print("Loading ", k[2:])
                d[k[2:]] = v
            else:
                miss = miss + 1
        print("Unable to find {} parameters!".format(miss))
    else:
        for k, v in file_dict.items():
            if k in model_dict.keys():
                # print("Loading ", k)
                d[k] = v
            else:
                miss = miss + 1
        print("Unable to find {} parameters!".format(miss))
    model_dict.update(d)
    model.load_state_dict(model_dict)
    return model
