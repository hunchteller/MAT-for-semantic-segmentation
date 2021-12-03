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
import matplotlib.pyplot as plt

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


def vis_net():
    transform = augmentations.trainAugmentation(512)

    v_transform = augmentations.valAugmentation(512)

    train_set = potsdam.ISPRS(root='../datasets_potsdam', transform=transform)
    val_set = potsdam.ISPRS(root='../datasets_potsdam', train=False, transform=v_transform)

    train_loader = DataLoader(train_set, batch_size=batch, num_workers=0, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch, num_workers=0, shuffle=False, drop_last=False)

    model = Seg_tran(
        num_classes=6,
        h=300,
        w=300,
        k1=5,
        k2=5,
        dim=512,
        depth=5,
        heads=16,
        dim_head=64,
        ratio=4,
    )

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.5))
    # crit = criterion.CrossEntropy()

    # def lr_sche(dd):
    #     if dd < 1000:
    #         rr = dd / 1000 * (lr - final_lr) / lr
    #     else:
    #         rr = (1 - dd / (epochs * len(train_loader))) ** 0.5 * (lr - final_lr) / lr
    #     return rr

    # sche = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sche)
    # t_loss = LossMeter()
    # v_loss = LossMeter()

    t_metric = metric.SegmentationMetric(6)
    v_metric = metric.SegmentationMetric(6)

    model.load_state_dict(torch.load('logs/ckpt/Potsdam seg, version 4. single label, downscale 4, size 300, poly 0.5, dim 512, aug_only mirror/100.pth'))
    if cuda:
        model.cuda()

    # for epoch in range(0, epochs):
    #     for ii, (img, label) in enumerate(train_loader):
    #         if cuda:
    #             img = img.float().cuda()
    #             label = label.cuda()
    #             # lbl2 = lbl2.cuda()
    #         optimizer.zero_grad()
    #         pred = model(img)
    #         # loss0 = crit(pred0, lbl2)
    #         loss = crit(pred, label)
    #         # loss = 0.5 * loss0 + loss1
    #         loss.backward()
    #         optimizer.step()
    #         sche.step()
    #         t_loss.add(loss.item(), img.size(0))
    #         with torch.no_grad():
    #             pred = F.interpolate(pred, size=(label.size(1), label.size(2)), mode='bilinear', align_corners=True)
    #             pred = pred.argmax(1).detach().cpu()
    #             t_metric.addBatch(pred, label.detach().cpu())
    # 
    #         if ii % 10 == 0:
    #             print(
    #                 f"[epoch {epoch}]\t [iter {ii}/{len(train_loader)}]\t [Loss {loss.item():.5f}]\t [lr {sche.get_last_lr()[0]:.3e}]")
    # 
    #     t_writer.add_scalar('loss', t_loss.avg(), global_step=epoch)
    #     t_writer.add_scalars('IOU', dict(zip(classes, t_metric.IntersectionOverUnion())), global_step=epoch)
    #     t_writer.add_scalars('F1', dict(zip(classes, t_metric.F1Score())), global_step=epoch)
    # 
    #     logger.info(f'[epoch {epoch}]\t [Loss {t_loss.avg():.5f}]\t')
    #     logger.info(f"[train {epoch}]\t [IoU {t_metric.IntersectionOverUnion()}]\t [F1 {t_metric.F1Score()}]\n")
    #     t_loss.reset()
    #     t_metric.reset()

        # if epoch % 5 == 4:
        #     torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{epochs}.pth"))

    model.eval()
    with torch.no_grad():
        for img, label in val_loader:
            if cuda:
                img = img.float().cuda()
                # label = label.cuda()
            pred = model(img)
            pred = F.interpolate(pred, size=(label.size(1), label.size(2)), mode='bilinear', align_corners=True)


            pred = pred.softmax(1)
            mask = pred.argmax(1).detach().cpu()

            acc = (mask == label).sum() / (label < 10).sum()

            plt.ion()
            plt.figure(1)
            plt.subplot(1, 3, 1)
            img = img[0].detach().cpu().permute(1, 2, 0).contiguous().numpy()
            plt.imshow(img/255)
            plt.title('ori')
            plt.subplot(1, 3, 2)
            plt.imshow(label[0], norm=plt.Normalize(0, 6), cmap=plt.cm.jet)
            plt.title('gt')
            plt.subplot(1, 3, 3)
            plt.imshow(mask[0], norm=plt.Normalize(0, 6), cmap=plt.cm.jet)
            plt.title(acc)
            plt.waitforbuttonpress(0)

            v_metric.reset()


if __name__ == "__main__":
    desp = 'Potsdam seg, version 4. single label, downscale 4, size 300, poly 0.5, dim 512, aug_only mirror'

    classes = ['Surfaces', 'Building', 'Low Veg', 'Tree', 'Car', 'Background']

    seed, batch, lr, final_lr, betas, epochs, cuda = 27, 1, 1e-4, 1e-6, (0.9, 0.999), 100, True
    set_seed(seed)
    # ckpt_dir, tb_dir, log_dir = os.path.join('logs', 'ckpt', desp), os.path.join('logs', 'tb', desp), os.path.join(
    #     'logs', 'log', desp)
    # os.makedirs(ckpt_dir, exist_ok=True)
    # os.makedirs(log_dir, exist_ok=True)
    #
    # t_writer = SummaryWriter(os.path.join(tb_dir, 'train'))
    # v_writer = SummaryWriter(os.path.join(tb_dir, 'test'))
    # logger = log.Log(f'{log_dir}/log.logging')
    # logger.info(f'seed {seed}, batch {batch}, lr {lr:.3e}, final_lr {final_lr:.3e}, betas {betas}, epochs {epochs}')

    vis_net()
