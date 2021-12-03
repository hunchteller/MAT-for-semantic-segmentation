import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from torch.utils.data import DataLoader
import argparse
import numpy as np
from utils import log, augmentations, potsdam, criterion, metric
import torchvision.datasets as ds
from model.seg_model3 import Seg_tran
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import random

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

def mixup(image):
    if random.random() > 0.5:
    # if random.random() > 2:
        inds = torch.randperm(image.size(0))
        alpha = random.random()
        image = alpha * image + (1 - alpha) * image[inds]
        return image, alpha, inds
    else:
        return image, None, None 

def train_net():
    # transform = augmentations.trainAugmentation(300)

    fmt = '{:^10}' * 7
    fmt_num = '{:^10}' + '{:^10.2f}' * 6

    transform = A.Compose([
        A.RandomResizedCrop(512, 512, scale=(0.8, 1.2), ratio=(1, 1)),
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    v_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])

    train_set = potsdam.ISPRS(root='../vaihingen_split', transform=transform)
    val_set = potsdam.ISPRS(root='../vaihingen_split', train=False, transform=v_transform)

    train_loader = DataLoader(train_set, batch_size=batch, num_workers=6, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch, num_workers=6, shuffle=False, drop_last=False)

    model = Seg_tran(
        num_classes=6,
        h=512,
        w=512,
        k1=8,
        k2=8,
        dim=256,
        depth=[2, 2, 1],
        heads=8,
        dim_head=32,
        ratio=4,
        attn_drop=0.5,
        proj_drop=0.5,
    )
    
    # model.load_state_dict(torch.load('logs/ckpt/49.pth')['params'])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=5e-6)
    # optimizer.load_state_dict(torch.load('logs/ckpt/49.pth', map_location='cuda')['optim'])
    # optimizer = optimizer.cuda()
    ce = criterion.CrossEntropy(weight = torch.tensor([1.0, 1, 2, 1, 2, 0.5], device='cuda'))
    dice = criterion.DiceLoss()

    def lr_sche(dd):
        if dd < 100:
            rr = dd / 1000 * (lr - final_lr) / lr
        else:
            rr = (1 - dd / (epochs * len(train_loader))) ** 0.9 * (lr - final_lr) / lr
        return rr

    sche = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sche)
    t_loss = LossMeter()

    t_metric = metric.SegmentationMetric(5)
    v_metric = metric.SegmentationMetric(5)

    if cuda:
        model.cuda()

    best_f1 = 0
    for epoch in range(0, epochs):
        model.train()
        for ii, (img, label) in enumerate(train_loader):
            if cuda:
                img = img.float().cuda()
                label = label.long().cuda()
                # lbl2 = lbl2.cuda()
            img, alpha, inds = mixup(img)

            optimizer.zero_grad()
            pred = model(img)
            if alpha is not None:
                loss = alpha * ce(pred, label) + (1-alpha) * ce(pred, label[inds])
            else:
                loss = ce(pred, label)
            loss.backward()
            optimizer.step()
            sche.step()
            t_loss.add(loss.item(), img.size(0))
            with torch.no_grad():
                pred = F.interpolate(pred, size=(label.size(1), label.size(2)), mode='bilinear', align_corners=True)
                pred = pred.argmax(1).detach().cpu()
                t_metric.addBatch(pred, label.detach().cpu())

            if ii % 10 == 0:
                print(
                    f"[epoch {epoch}]\t [iter {ii}/{len(train_loader)}]\t [Loss {loss.item():.5f}]\t [lr {sche.get_last_lr()[0]:.3e}]", flush=True)

        if (epoch +1 ) % 50 == 0:
            torch.save({'params': model.state_dict(), 'optim': optimizer.state_dict()},
                       os.path.join(ckpt_dir, f"{epoch}.pth"))

        if (epoch+1) % 5 ==0 :
            model.eval()
            with torch.no_grad():
                for img, label in val_loader:
                    if cuda:
                        img = img.float().cuda()
                        # label = label.cuda()
                    pred = model(img)
                    pred = F.interpolate(pred, size=(label.size(1), label.size(2)), mode='bilinear', align_corners=True)
                    v_metric.addBatch(pred.argmax(1).detach().cpu(), label)

            iou_v, f1_v = v_metric.IntersectionOverUnion()*100, v_metric.F1Score()*100
            iou_t, f1_t = t_metric.IntersectionOverUnion()*100, t_metric.F1Score()*100

            t_writer.add_scalar('loss', t_loss.avg(), global_step=epoch)
            t_writer.add_scalars('IOU', dict(zip(classes, iou_t)), global_step=epoch)
            t_writer.add_scalars('F1', dict(zip(classes, f1_t)), global_step=epoch)
            v_writer.add_scalars('IOU', dict(zip(classes, iou_v)), global_step=epoch)
            v_writer.add_scalars('F1', dict(zip(classes, f1_v)), global_step=epoch)

            logger.info(fmt.format(f'epoch {epoch}', *classes, 'average'))
            logger.info(fmt_num.format('Train IoU', *iou_t, iou_t.mean()))
            logger.info(fmt_num.format('Val IoU', *iou_v, iou_v.mean()))
            logger.info(fmt_num.format('Train F1', *f1_t, f1_t.mean()))
            logger.info(fmt_num.format('Val F1', *f1_v, f1_v.mean()))
            if f1_v.mean() > best_f1:
                torch.save({'params': model.state_dict(), 'optim': optimizer.state_dict()},
                       os.path.join(ckpt_dir, f"best.pth"))
                best_f1 = f1_v.mean()
            v_metric.reset()
        t_loss.reset()
        t_metric.reset()
    print(best_f1)

if __name__ == "__main__":
    desp = 'vai 1_weight_decay_drop_0.5'

    classes = ['Surfaces', 'Building', 'Low Veg', 'Tree', 'Car', 'Clutter']

    seed, batch, lr, final_lr, betas, epochs, alpha, cuda = 27, 4, 1e-4, 5e-6, (0.9, 0.999), 600, 0.5, True
    set_seed(seed)
    ckpt_dir, tb_dir, log_dir = os.path.join('logs', 'ckpt'), os.path.join('logs', 'tb', desp), os.path.join(
        'logs', 'log', desp)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    t_writer = SummaryWriter(os.path.join(tb_dir, 'train'))
    v_writer = SummaryWriter(os.path.join(tb_dir, 'test'))
    logger = log.Log(f'{log_dir}/log.txt')
    logger.info(
        f'{desp} ;seed {seed}, batch {batch}, lr {lr:.3e}, final_lr {final_lr:.3e}, betas {betas}, epochs {epochs}')

    train_net()
