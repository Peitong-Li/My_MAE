from tkinter import E
import numpy
from PIL import Image, ImageFile

import numpy as np
from torch.utils.data import Dataset
from IPython import embed
import os.path as osp
import random
import torch
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, mask=None, root=''):
        self.dataset = dataset
        self.transform = transform
        self.mask = mask
        self.no_mask_num = 0
        self.root = root
        self.test_size = [128, 256]


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, _, _ = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.mask is not None:
            img_name = img_path.split("/")[-1].split('.')[0]
            mask_type = img_path.split("/")[-2]
            if mask_type == 'bounding_box_test':
                current_mask_path = self.root + '/gallery_mask_numpy/' + img_name + '.npy'
            else:
                current_mask_path = self.root + '/query_mask_numpy/' + img_name + '.npy'
            if current_mask_path in self.mask:
                mask = np.load(current_mask_path)
                mask_np = mask[:, :, 0] != 0
                mask = np.array(Image.fromarray(mask_np).resize((224, 224)))
            else:
                mask = np.zeros([224, 224])
            return img, pid, img_path, mask
        else:
            return img, pid