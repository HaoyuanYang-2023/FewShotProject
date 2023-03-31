import torch.nn as nn

from torchvision.models import resnet18


def conv_block(in_channels, out_channels):
    """
    returns a block conv-bn-relu-pool
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def res():
    model = resnet18(weights=True)
    model.fc = Identity()
    return model


class ProtoNet(nn.Module):
    """
    Model as described in the reference paper, source:
    https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models
    /few_shot.py#L62-L84
    """

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        '''self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )'''

        self.encoder = res()

    def forward(self, x):
        x = self.encoder(x)
        # return x.view(x.size(0), -1)
        return x
