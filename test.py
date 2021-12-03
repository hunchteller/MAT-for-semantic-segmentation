import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import argparse
import numpy as np
from utils import log, augmentations, potsdam, criterion, metric
import torchvision.datasets as ds
from model.seg_model3 import Seg_tran
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    return


class LossMeter():
    def __init__(self):
        self.loss = 0
        self.num = 0

    def add(self, loss, num):
        self.loss += loss * num
        self.num += num

    def avg(self):
        return self.loss / self.num

    def reset(self):
        self.loss = 0
        self.num = 0


def test_net():
    # transform = augmentations.trainAugmentation(300)
    #
    # transform = A.Compose([
    #     A.RandomCrop(width=300, height=300),
    #     # A.HorizontalFlip(p=0.5),
    #     # A.VerticalFlip(p=0.5),
    #     # A.Rotate(limit=)
    #     A.Normalize(),
    #     ToTensorV2(),
    # ])

    v_transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

    # train_set = potsdam.ISPRS(root='../datasets_potsdam', transform=transform)
    val_set = potsdam.ISPRS(root='../datasets_potsdam', train=False, transform=v_transform)

    # train_loader = DataLoader(train_set, batch_size=batch, num_workers=0, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch, num_workers=0, shuffle=False, drop_last=False)

    model = Seg_tran(
        num_classes=6,
        h=300,
        w=300,
        k1=5,
        k2=5,
        dim=256,
        depth=5,
        heads=8,
        dim_head=64,
        ratio=4,
    )

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.5))
    # crit = criterion.CrossEntropy()
    model.load_state_dict(torch.load('logs/ckpt/dbg04_2predhead, Potsdam seg, version 5. single label, downscale 4, size 300, poly 0.5, dim 256, bn/199.pth')['params'])


    v_metric = metric.SegmentationMetric(6)

    if cuda:
        model.cuda()


    model.eval()
    with torch.no_grad():
        for img, label in val_loader:
            if cuda:
                img = img.float().cuda()
                # label = label.cuda()
            pred, cls = model(img)
            pred = F.interpolate(pred, size=(label.size(1), label.size(2)), mode='bilinear', align_corners=True)
            cls = F.interpolate(cls, size=(label.size(1), label.size(2)), mode='bilinear', align_corners=True)
            pred = pred + 0.1 * cls
            v_metric.addBatch(pred.argmax(1).detach().cpu(), label)


    print(f"[val]\t [IoU {v_metric.IntersectionOverUnion()}]\t [F1 {v_metric.F1Score()}]\n")
    v_metric.reset()


if __name__ == "__main__":
    desp = 'dbg04_2predhead, Potsdam seg, version 5. single label, downscale 4, size 300, poly 0.5, dim 256, bn'

    classes = ['Surfaces', 'Building', 'Low Veg', 'Tree', 'Car', 'Background']

    seed, batch, lr, final_lr, betas, epochs, cuda = 27, 16, 1e-4, 1e-6, (0.9, 0.999), 200, True
    set_seed(seed)
    test_net()
