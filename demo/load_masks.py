# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         load_masks
# Author:       LPT
# Email:        lpt2820447@163.com
# Date:         2022/4/27 15:09
# Description:
# -------------------------------------------------------------------------------

import sys
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

from PIL import Image


dataset_name = 'Occluded_Duke'
dataset_path = 'D:/Datasets'
root = os.path.join(dataset_path, dataset_name)


if __name__ == '__main__':
    query_path = os.path.join(root, 'query_mask_numpy')
    masks_path = glob.glob(query_path + '/*.npy')
    masks_list = []
    for idx, mask in enumerate(masks_path):
        plt.rcParams['figure.figsize'] = [25, 25]
        plt.subplot(1, 2, 1)
        mask_np = np.load(mask)
        plt.imshow(mask_np)
        mask_np = mask_np[:, :, 0] != 0
        mask = Image.fromarray(mask_np).resize((224, 224))
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(mask))

        plt.show()
        # Image.fromarray(np.uint8(mask_np)*255).show()
        masks_list.append(mask_np)
    # query_mask_np = np.array(masks_list)