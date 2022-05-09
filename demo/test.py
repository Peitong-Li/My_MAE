# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         test
# Author:       LPT
# Email:        lpt2820447@163.com
# Date:         2022/4/30 10:46
# Description:
# -------------------------------------------------------------------------------

import sys
from IPython import embed
import numpy as np
import os
import torch
import argparse
import time
from PIL import Image
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torchvision import transforms as T
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from config import Config
import models_mae
from datasets.bases import ImageDataset
from datasets.dukemtmcreid import DukeMTMCreID
from datasets.market1501 import Market1501
from datasets.occ_duke import OCC_DukeMTMCreID


__factory = {
    'market1501': Market1501,
    'dukereidmtmc': DukeMTMCreID,
    'Occluded_Duke': OCC_DukeMTMCreID,
}

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids, img_paths, mask = zip(*batch)
    if mask[0] is not None:
        mask = torch.tensor(mask, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, img_paths, mask


def make_dataloader(args):
    # 图像预处理
    img_transform = T.Compose([
        T.Resize(Config.image_size, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # transform_train = T.Compose([
    #     T.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = __factory[Config.dataset_name](root=Config.root)
    train_num_classes = dataset.num_train_pids
    val_img_num = len(dataset.query)
    train_set = ImageDataset(dataset.train, img_transform)
    val_set = ImageDataset(dataset.query + dataset.gallery, transform=img_transform, root=dataset.dataset_dir, mask=dataset.query_mask + dataset.gallery_mask)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(train_set)

    train_Loader = DataLoader(
        dataset.train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=train_collate_fn,
    )

    val_Loader = DataLoader(dataset=val_set, batch_size=Config.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=args.pin_mem, collate_fn=val_collate_fn)

    Config.num_classes = train_num_classes
    return train_Loader, val_Loader, val_img_num, train_num_classes

def new_show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    image = torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    image = Image.fromarray(np.uint8(image.numpy())).resize((128, 256))
    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def run_batch_image(img, model, img_name):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [25, 25]

    plt.subplot(1, 4, 1)
    new_show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    new_show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    new_show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    new_show_image(im_paste[0], "reconstruction + visible")

    # plt.show()
    plt.savefig(f'./result/{img_name}.jpg')

def visual(model, loss, y, mask, x, n_iter):
    batch = x.shape[0]
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x).detach().cpu()

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [25, 25]
    sum = 1
    for i in range(batch):
        plt.subplot(batch, 4, sum)
        sum += 1
        new_show_image(x[i], "original")

        plt.subplot(batch, 4, sum)
        sum += 1
        new_show_image(im_masked[i], "masked")

        plt.subplot(batch, 4, sum)
        sum += 1
        new_show_image(y[i], "reconstruction")

        plt.subplot(batch, 4, sum)
        sum += 1
        new_show_image(im_paste[i], "reconstruction + visible")
    # plt.show()
    plt.savefig(f'./result/{str(n_iter)}.jpg')


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    global_rank = misc.get_rank()

    train_Loader, val_Loader, val_img_num, train_num_classes = make_dataloader(args)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
        # build model
        model = getattr(models_mae, arch)()
        # load model
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        return model

    chkpt_dir = 'mae_visualize_vit_large.pth'  # or 'mae_visualize_vit_large_ganloss.pth'
    model = prepare_model(chkpt_dir, 'mae_vit_large_patch16')

    model.to(device)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    start_time = time.time()

    for n_iter, (imgs, pids, imgpaths, masks) in enumerate(val_Loader):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device)
        loss, y, mask = model(imgs, mask_ratio=0.75, masks=masks)
        visual(model, loss, y, mask, imgs, n_iter)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
