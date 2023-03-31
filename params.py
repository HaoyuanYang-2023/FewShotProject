import argparse
import json


def get_params(fromjson=False, json_file=None):
    if not fromjson:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset',
                            type=str,
                            help='dataset name',
                            default='miniImageNet',
                            choices=['miniImageNet', 'tieredImageNet'])

        parser.add_argument('--train_root',
                            type=str,
                            help='path to dataset',
                            default='F:/miniImageNet/train')

        parser.add_argument('--val_root',
                            type=str,
                            help='path to dataset',
                            default='F:/miniImageNet/val')

        parser.add_argument('--test_root',
                            type=str,
                            help='test set path',
                            default='F:/miniImageNet/test')

        parser.add_argument('--model',
                            type=str,
                            help='model to use',
                            default="BDCResNet12",
                            choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101',
                                     'ResNet152', 'BDCResNet12'])

        parser.add_argument('--pretrain_model_path',
                            type=str,
                            help='path to saved pretrained model',
                            default=None)

        parser.add_argument('--meta_model_path',
                            type=str,
                            help='path to saved meta-trained model',
                            default=None)

        parser.add_argument('--method',
                            type=str,
                            choices=["protonet"],
                            default="protonet")

        parser.add_argument('--reduced_dim',
                            type=int,
                            default=256,
                            help="Dimensions to reduce before cov layer")

        parser.add_argument('--exp',
                            type=str,
                            help='exp information',
                            default='exp1')

        parser.add_argument('--epochs',
                            type=int,
                            help='number of epochs to train for, default=100',
                            default=100)

        parser.add_argument('--lr',
                            type=float,
                            help='learning rate for the model, default=0.001',
                            default=1e-3)

        parser.add_argument('--optimizer',
                            type=str,
                            default='Adam',
                            choices=['Adam', 'SGD'],
                            help="Optimizers")

        parser.add_argument('--momentum',
                            type=float,
                            default=0.9,
                            help='momentum')

        parser.add_argument('--nesterov',
                            type=bool,
                            default=True,
                            help='nesterov momentum')

        parser.add_argument('--weight_decay',
                            type=float,
                            default=5e-4,
                            help='weight decay (default: 5e-4)')

        parser.add_argument('--lrS',
                            type=int,
                            help='StepLR learning rate scheduler step, default=40',
                            default=40)

        parser.add_argument('--lrG',
                            type=float,
                            help='StepLR learning rate scheduler gamma, default=0.5',
                            default=0.5)

        parser.add_argument('--episodes',
                            type=int,
                            help='number of episodes per epoch, default=100',
                            default=100)

        parser.add_argument('--n_way',
                            type=int,
                            help='number of random classes per episode for training, default=5',
                            default=5)

        parser.add_argument('--n_support',
                            type=int,
                            help='number of samples per class to use as support for training, default=5',
                            default=5)

        parser.add_argument('--n_query',
                            type=int,
                            help='number of samples per class to use as query for training, default=16',
                            default=16)

        parser.add_argument('--print_freq',
                            type=int,
                            help="Step interval to print",
                            default=100)

        parser.add_argument('-seed', '--manual_seed',
                            type=int,
                            help='input for the manual seeds initializations',
                            default=7)

        parser.add_argument('--gpu', type=int,
                            default=0,
                            help='enables cuda')

        parser.add_argument('--resume',
                            action='store_true',
                            help='resume training')

        parser.set_defaults(augment=True)

        args = parser.parse_args()

        return args

    else:
        with open(json_file) as f:
            summary_dict = json.load(fp=f)

        '''for key in summary_dict.keys():
            args[key] = summary_dict[key]'''
        args = argparse.Namespace(**summary_dict)
        return args
