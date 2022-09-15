import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class Figaro1kDataset(Dataset):
    def __init__(self, imgdir, segdir, transform=None):
        self.img_dir = imgdir
        self.img_files = sorted(os.listdir(imgdir))
        self.seg_dir = segdir
        self.seg_files = sorted(os.listdir(segdir))
        assert len(self.img_files) == len(self.seg_files)
        self.transform = transform
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        with Image.open(os.path.join(self.img_dir, self.img_files[idx])) as img, \
             Image.open(os.path.join(self.seg_dir, self.seg_files[idx])) as seg:
            img = img.convert('RGB')
            seg = seg.convert('L')
        
        if self.transform is not None:
            '''        
            print('img.size : {}'.format(img.size));    #     RGB exit(0)           #   (1000, 1000)
            print('seg.size : {}'.format(seg.size));    #     RGB exit(0)           #   (1000, 1000)
            print('img.getextrema() : {}'.format(img.getextrema()));    #exit(0)    #   (0, 255), (0, 255), (0, 255)
            print('seg.getextrema() : {}'.format(seg.getextrema()));    #exit(0)    #   (0, 255)
            '''
            img, seg = self.transform(img, seg)
            '''
            print('img.shape : {}'.format(img.shape));    #     RGB exit(0)         #   [3, 512, 512]
            print('seg.shape : {}'.format(seg.shape));    #     RGB exit(0)         #   [1, 512, 512]
            print('torch.max(img) : {}'.format(torch.max(img)));    #exit(0)        #   1.0
            print('torch.max(seg) : {}'.format(torch.max(seg)));    exit(0)         #   1.0
            '''

        return img, seg
