import argparse
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--tar_dir', type=str)
parser.add_argument('--phase', type=str)
args = parser.parse_args()


def untarring(phase):
    if args.tar_dir is None:
        raise ValueError("tar_dir must be not None")
    print('Untarring ILSVRC2012 ' + phase + ' package')
    imagenet_dir = './ImageNet/' + phase
    if not os.path.exists(imagenet_dir):
        os.mkdir(imagenet_dir)
    os.system('tar xvf ' + str(args.tar_dir) + ' -C ' + imagenet_dir)
    return imagenet_dir


if __name__ == "__main__":
    untarring(args.phase)
