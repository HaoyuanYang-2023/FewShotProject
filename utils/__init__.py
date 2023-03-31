import torch
import numpy as np
import random

def init_seed(manual_seed):
    """
    Disable cudnn to maximize reproducibility
    """
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_lr_scheduler(optim,lrG,milestones):
    """
    Initialize the learning rate scheduler
    """
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optim,
                                                gamma=lrG,
                                                milestones=milestones)
