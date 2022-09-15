import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class CelebAMaskDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transforms=None):
        """
        Args:
            root_dir (str): root directory of dataset
            joint_transforms (torchvision.transforms.Compose): tranformation on both data and target
            image_transforms (torchvision.transforms.Compose): tranformation only on data
            mask_transforms (torchvision.transforms.Compose): tranformation only on target
            gray_image (bool): True if to add gray images
        """

        txt_file = 'train_val_list.txt' if is_train else 'test_list.txt'
        path_txt = os.path.join(root_dir, txt_file)
        name_list = CelebAMaskDataset.parse_name_list(path_txt)
        fn_li_ignore = os.path.join(root_dir, 'li_ignore_due_to_intentional_reflection.txt')
        path_li_ignore = os.path.join(root_dir, fn_li_ignore)
        #print('path_li_ignore :', path_li_ignore);  exit(0)
        if os.path.exists(path_li_ignore):
            li_name_ignore = CelebAMaskDataset.parse_name_list(path_li_ignore)
            print('len(li_name_ignore) :', len(li_name_ignore));   #   591    #exit(0)
            if len(li_name_ignore):
                print('len(name_list) b4 :', len(name_list))        #   27176
                name_list = [naim for naim in name_list if not (naim in li_name_ignore)] 
                print('len(name_list) after :', len(name_list));    #   26628   #   exit(0)
        mask_dir = os.path.join(root_dir, 'CelebAMask-HQ-mask-anno')
        img_dir = os.path.join(root_dir, 'CelebA-HQ-img')
        li_name_hair = []
        self.mask_path_list = [];   self.img_path_list = []
        for naim in name_list:
            #print('naim :', naim)
            id_num = int(naim)
            folder_num = id_num // 2000
            fn_mask_hair = os.path.join(mask_dir, str(folder_num), str(id_num).rjust(5, '0') + '_hair.png')
            #print('fn_mask_hair :', fn_mask_hair);  #   /data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_hair.png    #exit(0)
            if os.path.exists(fn_mask_hair):
                self.mask_path_list.append(fn_mask_hair)
                self.img_path_list.append(os.path.join(img_dir, str(id_num) + '.jpg'))
                #print('self.img_path_list :', self.img_path_list);    #exit(0)
                #print('self.mask_path_list :', self.mask_path_list);    exit(0)
                li_name_hair.append(naim)
        #print('len(li_name_hair) :', len(li_name_hair), ', len(name_list) :', len(name_list));  #   26001, 26628    #exit(0);
        if len(li_name_hair) != len(name_list):
            name_list = li_name_hair
        #print('self.img_path_list :', self.img_path_list);    exit(0)
        #print('self.mask_path_list :', self.mask_path_list);    exit(0)

        #self.img_path_list = [os.path.join(img_dir, naim + '.jpg') for naim in name_list]
        #self.mask_path_list = [os.path.join(mask_dir, str(int(naim) // 2000), naim.just(5, '0'), + '_hair.png') for naim in name_list]
        self.transforms = transforms
        ''' 
        self.joint_transforms = joint_transforms
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.gray_image = gray_image
        '''

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert('RGB')
        mask_path = self.mask_path_list[idx]
        mask = Image.open(mask_path).convert('L')
       
        img = img.resize(mask.size, Image.ANTIALIAS)

        #mask = CelebDataset.rgb2binary(mask)
       
        if self.transforms is not None:
            '''
            print('img.size : {}'.format(img.size));    #     RGB exit(0)           (512, 512)
            print('mask.size : {}'.format(mask.size));    #     RGB exit(0)         (512, 512)
            print('img.getextrema() : {}'.format(img.getextrema()));    #exit(0)    ((0, 253), (0, 237), (0, 242))
            print('mask.getextrema() : {}'.format(mask.getextrema()));    #exit(0)  (0, 255)
            '''
            img, mask = self.transforms(img, mask)
            '''
            print('img.shape : {}'.format(img.shape));    #     RGB exit(0)         [3, 512, 512]
            print('mask.shape : {}'.format(mask.shape));    #     RGB exit(0)       [1, 512, 512]
            print('torch.max(img) : {}'.format(torch.max(img)));    #exit(0)        0.996
            print('torch.max(mask) : {}'.format(torch.max(mask)));    exit(0)       1.0
            '''
        return img, mask

        '''
        if self.gray_image:
            gray = img.convert('L')
            gray = np.array(gray,dtype=np.float32)[np.newaxis,] / 255
            #return img, mask, gray

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        
        if self.gray_image:
            gray = img.convert('L')
            gray = np.array(gray,dtype=np.float32)[np.newaxis,]/255
            return img, mask, gray, img_path, mask_path 
        else:
            return img, mask, img_path, mask_path
        '''

    def __len__(self):
        return len(self.mask_path_list)

    @staticmethod
    def rgb2binary(mask):
        """transforms RGB mask image to binary hair mask image.
        """
        mask_arr = np.array(mask)
        mask_map = mask_arr == np.array([255, 0, 0])
        mask_map = np.all(mask_map, axis=2).astype(np.float32)
        return Image.fromarray(mask_map)

    @staticmethod
    def parse_name_list(fp):
        lines = None
        with open(fp, 'r') as fin:
            lines = fin.readlines()
        return lines
        '''    
        parsed = list()
        for line in lines:
            name, num = line.strip().split(' ')
            num = format(num, '0>4')
            filename = '{}_{}'.format(name, num)
            parsed.append((name, filename))
        return parsed
        '''


'''
class CelebAMaskDataset(Dataset):
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
            img, seg = self.transform(img, seg)
            
        return img, seg
'''        
