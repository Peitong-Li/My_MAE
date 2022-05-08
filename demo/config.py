# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         config
# Author:       LPT
# Email:        lpt2820447@163.com
# Date:         2022/4/22 21:14
# Description:
# -------------------------------------------------------------------------------

import sys
from IPython import embed
import numpy as np
import os


class Config():
    dataset_name = 'Occluded_Duke'
    root = '/home/worker/User/LPT/datasets'
    # root = 'D://Datasets'
    num_workers = 0
    num_classes = 1000
    image_size = [224, 224]  # 输入和输出的图片尺寸
    batch_size = 8
    max_epoch = 4000
    save_img_path = './generate_img'  # 生成的图片保存路径
