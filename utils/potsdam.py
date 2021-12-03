from torch.utils.data import Dataset
import torch
import glob
import os.path as osp
import cv2
import numpy as np
import random

class ISPRS(Dataset):
    def __init__(self, root, train=True, transform=None, cutmix=False):
        if train:
            self.img_dir = osp.join(root, 'train_imgs_split')
            self.lbl_dir = osp.join(root, 'train_gray_split')
            self.imglist = glob.glob(osp.join(self.img_dir, '*.png'))

        else:
            self.img_dir = osp.join(root, 'test_imgs_split')
            self.lbl_dir = osp.join(root, 'test_gray_split')
            self.imglist = glob.glob(osp.join(self.img_dir, '*.png'))
        # print(self.imglist)
        self.transform = transform
        self.cutmix = cutmix


    def __len__(self):
        return len(self.imglist)
    
    def rand_bbox(self, size, lam):
        H, W, C = size
        cut_rat = np.sqrt(1-lam)
        cut_w = np.int(W*cut_rat)
        cut_h = np.int(H*cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    
    def __getitem__(self, index):
        img = cv2.imread(self.imglist[index]).astype(np.float32)
        lbl_name = osp.join(self.lbl_dir, self.imglist[index].split('/')[-1].replace('.jpg', '.png'))
        # print(lbl_name)
        lbl = cv2.imread(lbl_name, cv2.IMREAD_GRAYSCALE)
        # print(lbl)
        if self.cutmix:
            print('cutmix mode is on')
            r = random.random()
            if r>0.5:
                lam = np.random.beta(1, 1)
                rand_index = random.choice(range(len(self.imglist)))
                img2 = cv2.imread(self.imglist[rand_index]).astype(np.float32)
                lbl_name2 = osp.join(self.lbl_dir, self.imglist[rand_index].split('/')[-1].replace('.jpg', '.png'))
                lbl2 = cv2.imread(lbl_name2, cv2.IMREAD_GRAYSCALE)

                bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.shape, lam)
                img[bby1:bby2, bbx1:bbx2] = img2[bby1:bby2, bbx1:bbx2]
                lbl[bby1:bby2, bbx1:bbx2] = lbl2[bby1:bby2, bbx1:bbx2]

        if self.transform:
            transformed = self.transform(image=img, mask=lbl)
            img, lbl = transformed['image'], transformed['mask']
        return img, lbl


if __name__ == '__main__':
    sets = ISPRS('../datasets_potsdam')
    print(sets.__getitem__(1))


