from torch.utils.data import Dataset
import torch
import glob
import os.path as osp
import cv2
import numpy as np
import os


class ISPRS(Dataset):
    def __init__(self, root, train=True, transform=None):
        if train:
            self.img_dir = osp.join(root, 'train_images')
            self.lbl_dir = osp.join(root, 'train_segmentationRaw')
            self.imglist = glob.glob(osp.join(self.img_dir, '*.jpg'))

        else:
            self.img_dir = osp.join(root, 'test_images')
            self.lbl_dir = osp.join(root, 'test_segs_gray')
            self.imglist = glob.glob(osp.join(self.img_dir, '*.tif'))
        # print(self.imglist)
        self.transform = transform
        # print(self.imglist)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        img = cv2.imread(self.imglist[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        lbl_name = osp.join(self.lbl_dir, os.path.split(self.imglist[index])[-1].replace('.tif', '_noBoundary.png'))
        lbl = cv2.imread(lbl_name, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            transformed = self.transform(image=img, mask=lbl)
            img, lbl = transformed['image'], transformed['mask']
        return img, lbl


if __name__ == '__main__':
    sets = ISPRS('../../datasets_potsdam', train=False)
    print(sets.__getitem__(1))
