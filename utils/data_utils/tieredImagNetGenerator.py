"""
tieredImageNet splits followed
Ren, Mengye et al. “Meta-Learning for Semi-Supervised Few-Shot Classification.” ArXiv abs/1803.00676 (2018): n. pag.
https://github.com/renmengye/few-shot-ssl-public/tree/master/fewshot/data/tiered_imagenet_split
"""
import argparse
import os
import pandas as pd
import cv2
from tqdm import tqdm

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--tar_dir', type=str)
parser.add_argument('--imagenet_dir', type=str, default='F:/ILSVRC2012/train')
parser.add_argument('--tieredImageNet_dir', type=str, default='F:/tieredImageNet')
parser.add_argument('--image_resize', type=int, default=84)

args = parser.parse_args()


def get_class_list(split='train'):
    filename = 'F:/FewShotProject/split_csv/tieredImageNet/' + split + '.csv'
    csv = pd.read_csv(filename, delimiter=',')
    category_list = []
    class_list = []
    for row in csv.values:
        category_list.append(row[1])
        class_list.append(row[0])
    category_list = list(set(category_list))
    return class_list, category_list


class tieredImageNetGenerator(object):
    def __init__(self, input_args):
        self.input_args = input_args
        self.imagenet_dir = input_args.imagenet_dir
        self.tiered_imagenet_dir = input_args.tieredImageNet_dir
        if not os.path.exists(self.tiered_imagenet_dir):
            os.mkdir(self.tiered_imagenet_dir)
        self.image_resize = self.input_args.image_resize

        self.train_class_list, _ = get_class_list('train')
        self.val_class_list, _ = get_class_list('val')
        self.test_class_list, _ = get_class_list('test')

        self.all_class_list = self.train_class_list + self.val_class_list + self.test_class_list

    def link_imagenet(self):
        images_keys = self.all_class_list
        target_base = self.tiered_imagenet_dir

        # self.process_splits('train', target_base)
        self.process_splits('test', target_base)
        self.process_splits('val', target_base)

    def process_splits(self, split, target_base):
        if split not in ['train', 'val', 'test']:
            raise ValueError("Value of phase should be in [train, val, test]")

        if split == 'train':
            class_list = self.train_class_list
        if split == 'val':
            class_list = self.val_class_list
        if split == 'test':
            class_list = self.test_class_list

        target_dir = os.path.join(target_base, split)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        print('Process ' + split + ' data...')
        for i, keys in enumerate(class_list):
            print('Process class ' + keys)
            this_class_target_dir = target_dir + '/' + keys + '/'
            if not os.path.exists(this_class_target_dir):
                os.mkdir(this_class_target_dir)
            this_class_tar_dir = os.path.join(self.imagenet_dir, keys)
            os.system('tar xf ' + this_class_tar_dir + '.tar' + ' -C ' + this_class_target_dir)
            image_names = os.listdir(this_class_target_dir)
            for idx, name in tqdm(enumerate(image_names)):
                path = os.path.join(this_class_target_dir, name)
                im = cv2.imread(path)
                im_resized = cv2.resize(im, (self.image_resize, self.image_resize), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path, im_resized)


if __name__ == "__main__":
    dataset_generator = tieredImageNetGenerator(args)
    dataset_generator.link_imagenet()
