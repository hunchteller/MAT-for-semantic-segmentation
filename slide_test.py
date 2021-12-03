import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from utils import potsdam_test, metric
from model.seg_model3 import Seg_tran
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# from tqdm import tqdm

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



def slide_inference():
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """
    v_transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ])


    # train_set = potsdam.ISPRS(root='../datasets_potsdam', transform=transform)
    val_set = potsdam_test.ISPRS(root='../vaihingen_split', train=False, transform=v_transform)

    # train_loader = DataLoader(train_set, batch_size=batch, num_workers=0, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch, num_workers=0, shuffle=False, drop_last=False)

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
        attn_drop=0.3,
        proj_drop=0.3,
    )

    model.load_state_dict(torch.load('logs/ckpt/599.pth')['params'])


    v_metric = metric.SegmentationMetric(5)

    if cuda:
        model.cuda()


    model.eval()
    with torch.no_grad():
        for img, label in val_loader:
            if cuda:
                img = img.cuda()
            batch_size, _, h_img, w_img = img.size()
            num_classes = 6
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            preds = torch.zeros((batch_size, num_classes, h_img, w_img))
            count_mat = torch.zeros((batch_size, 1, h_img, w_img))
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    print(f'{h_idx}/{h_grids}, {w_idx}/{w_grids}')
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = min(y2 - h_crop, y1)
                    x1 = min(x2 - w_crop, x1)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    pred = model(crop_img)
                    pred = F.interpolate(pred, size=(h_crop, w_crop), mode='bilinear', align_corners=True)

                    preds += F.pad(pred.detach().cpu(),
                                   (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))
                    # print(preds.shape, temp.shape)
                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            v_metric.addBatch(preds.argmax(1).detach().cpu(), label)
        iou = v_metric.IntersectionOverUnion()
        f1 = v_metric.F1Score()
        print(f"[val]\t [IoU {*iou, iou.mean()}]\t [F1 {*f1, f1.mean()}]\n")
        v_metric.reset()

if __name__ == "__main__":
    desp = 'dbg04_2predhead, Potsdam seg, version 5. single label, downscale 4, size 300, poly 0.5, dim 256, bn'

    classes = ['Surfaces', 'Building', 'Low Veg', 'Tree', 'Car', 'Background']

    seed, batch, lr, final_lr, betas, epochs, cuda = 27, 1, 1e-4, 1e-6, (0.9, 0.999), 200, True
    h_stride, w_stride = 200, 200
    h_crop, w_crop = 512, 512

    set_seed(seed)
    slide_inference()
