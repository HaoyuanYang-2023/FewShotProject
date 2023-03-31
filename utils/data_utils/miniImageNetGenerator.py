import argparse
import os
import numpy as np
import pandas as pd
import glob
import cv2
from shutil import copyfile
from tqdm import tqdm

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', type=str, choices=['train', 'val'])
parser.add_argument('--imagenet_dir', type=str)
parser.add_argument('--miniImageNet_dir', type=str)
parser.add_argument('--split_filepath', typr=str)
parser.add_argument('--image_resize', type=int, default=84)

args = parser.parse_args()


def get_mini_keys():
    split_lists = ['train', 'val', 'test']
    keys = []
    for i in split_lists:
        csv_path = os.path.join('F:/FewShotProject/split_csv/miniImageNet', i) + '.csv'
        label = pd.read_csv(csv_path).get('label')
        label = np.unique(np.array(label)).tolist()
        keys += label
    return keys


class MiniImageNetGenerator(object):
    def __init__(self, input_args):
        self.processed_img_dir = './miniImageNet'
        self.mini_keys = get_mini_keys()
        self.input_args = input_args
        self.imagenet_dir = input_args.imagenet_dir
        self.raw_mini_dir = './miniImageNet_raw'
        self.csv_paths = input_args.split_filepath
        if not os.path.exists(self.raw_mini_dir):
            os.mkdir(self.raw_mini_dir)
        self.image_resize = self.input_args.image_resize

    def untar_mini(self):
        for idx, keys in enumerate(self.mini_keys):
            print('Untarring ' + keys)
            os.system('tar xvf ' + self.imagenet_dir + '/' + keys + '.tar -C ' + self.raw_mini_dir)
        print('All the tar files are untarred')

    def process_original_files(self):
        split_lists = ['train', 'val', 'test']

        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)

        for this_split in split_lists:
            filename = os.path.join(self.csv_paths, this_split + '.csv')
            this_split_dir = self.processed_img_dir + '/' + this_split
            if not os.path.exists(this_split_dir):
                os.makedirs(this_split_dir)
            with open(filename) as csvfile:
                csv = pd.read_csv(csvfile, delimiter=',')
                images = {}
                print('Reading IDs....')

                for row in csv.values:
                    if row[1] in images.keys():
                        images[row[1]].append(row[0])
                    else:
                        images[row[1]] = [row[0]]

                print('Writing photos....')
                for cls in tqdm(images.keys()):
                    this_cls_dir = this_split_dir + '/' + cls
                    if not os.path.exists(this_cls_dir):
                        os.makedirs(this_cls_dir)
                    # find files which name matches '.../...cls...'
                    lst_files = glob.glob(self.raw_mini_dir + "/*" + cls + "*")
                    # sort file names, get index
                    lst_index = [int(i[i.rfind('_') + 1:i.rfind('.')]) for i in lst_files]
                    index_sorted = np.argsort(np.array(lst_index))
                    # get file names in miniImageNet, the name in csv indicates the file index in miniImageNet class
                    index_selected = [int(i[i.index('.') - 4:i.index('.')]) for i in images[cls]]
                    # note that names in csv begin from 1 not 0, get selected images indexes
                    selected_images = index_sorted[np.array(index_selected) - 1]
                    for i in np.arange(len(selected_images)):
                        if self.image_resize == 0:
                            copyfile(lst_files[selected_images[i]], os.path.join(this_cls_dir, images[cls][i]))
                        else:
                            im = cv2.imread(lst_files[selected_images[i]])
                            im_resized = cv2.resize(im, (self.image_resize, self.image_resize),
                                                    interpolation=cv2.INTER_AREA)
                            cv2.imwrite(os.path.join(this_cls_dir, images[cls][i]), im_resized)


if __name__ == "__main__":
    dataset_generator = MiniImageNetGenerator(args)
    dataset_generator.untar_mini()
    dataset_generator.process_original_files()
