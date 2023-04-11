#import av
#import pims
import os
import numpy as np
import torch
import shutil
import cv2
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image

def is_this_empty_string(strin):
    return (strin in (None, '')) or (not strin.strip())

def get_list_of_file_path_under_1st_with_3rd_extension(direc, include_subdirectories, ext = ''):
    li_path_total = []
    is_extension_given = not (is_this_empty_string(ext))
    if include_subdirectories:
        for dirpath, dirnames, filenames in os.walk(os.path.expanduser(direc)):
            n_file_1 = len(filenames)
            if n_file_1:
                if is_extension_given:
                    li_path = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(ext.lower())]
                else:
                    li_path = [os.path.join(dirpath, f) for f in filenames]
                n_file_2 = len(li_path)
                if n_file_2:
                    li_path_total += li_path
    else:
        for name_file_dir in os.listdir(direc):
            path_file_dir = os.path.join(direc, name_file_dir)
            if os.path.isfile(path_file_dir):
                if is_extension_given:
                    if name_file_dir.lower().endswith(ext.lower()):
                        li_path_total.append(path_file_dir)
                else:
                    li_path_total.append(path_file_dir)
    return sorted(li_path_total)


def remove_directory(direc):
    shutil.rmtree(direc)

def is_this_existing_directory(path_dir):
    return os.path.exists(path_dir) and os.path.isdir(path_dir)

def rm_directory_if_exist(direc):
    if is_this_existing_directory(direc):
        remove_directory(direc)    

def rm_and_mkdir(direc):
    rm_directory_if_exist(direc)
    os.makedirs(direc)

def tensor_0_1_to_ndarray_0_255(tenser):
    t3 = tenser.permute(1, 2, 0)
    t4 = t3.cpu().detach().numpy()
    t5 = 255 * t4
    #t6 = np.uint8(t5)
    return np.uint8(t5)
    #t7 = cv2.cvtColor(t6, cv2.COLOR_BGR2RGB)

def add_image_text(img, text, bgr_txt):
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    textX = (img.shape[1] - textsize[0]) // 2
    textY = textsize[1] + 10
    cv2.putText(img, text, (int(textX), int(textY)), font, 1, bgr_txt, 2, cv2.LINE_AA)
    return img

def compute_layout(li_hwc, is_horizontal, n_max_per_row_or_col, pxl_max_per_row_or_col, wh_interval):
    
    li_li_idx = []; li_li_xywh = [] 
    if is_horizontal:
        x_cur = 0;  y_cur = 0;   w_max = 0; h_max = 0;  w_sum = 0;  h_max_cur = -1
        li_idx = [];    li_xywh = []   
        for idx, hwc in enumerate(li_hwc):
            #print('idx :', idx) 
            h_cur, w_cur = hwc[0], hwc[1]
            xywh = (x_cur, y_cur, w_cur, h_cur)
            #print('xywh :', xywh)
            is_over_pixel = pxl_max_per_row_or_col > 0 and (w_sum + w_cur) > pxl_max_per_row_or_col
            is_over_slot = n_max_per_row_or_col > 0 and (len(li_idx) + 1) > n_max_per_row_or_col
            if is_over_pixel or is_over_slot:
                w_sum -= wh_interval[0]
                w_max = max(w_max, w_sum) 
                h_max += h_max_cur + wh_interval[1]
                li_li_idx.append(li_idx)
                li_li_xywh.append(li_xywh)
                x_cur = 0;  y_cur = h_max  
                xywh = (x_cur, y_cur, w_cur, h_cur)
                li_idx = [idx];    li_xywh = [xywh];  
                w_sum = w_cur + wh_interval[0];  h_max_cur = -1; 
                x_cur = w_sum
            else: 
                w_sum += w_cur + wh_interval[0]
                li_idx.append(idx)
                li_xywh.append(xywh)
                x_cur = w_sum;  
            h_max_cur = max(h_max_cur, h_cur)
        if 0 == h_max and 0 == w_max:
            w_max = w_sum;  
        li_li_idx.append(li_idx);   li_li_xywh.append(li_xywh) 
        h_max += h_max_cur
        w_max -= wh_interval[0];    #h_max -= wh_interval[1]        
    else:
        x_cur = 0;  y_cur = 0;  w_max = 0;  h_max = 0;  h_sum = 0;  w_max_cur = -1
        li_idx = []; li_xywh = []
        for idx, hwc in enumerate(li_hwc):
            #print('idx :', idx) 
            h_cur, w_cur = hwc[0], hwc[1]
            xywh = (x_cur, y_cur, w_cur, h_cur)
            #print('xywh :', xywh)
            is_over_pixel = pxl_max_per_row_or_col > 0 and (h_sum + h_cur) > pxl_max_per_row_or_col
            is_over_slot = n_max_per_row_or_col > 0 and (len(li_idx) + 1) > n_max_per_row_or_col
            if is_over_pixel or is_over_slot:
                h_sum -= wh_interval[1]
                h_max = max(h_max, h_sum)
                w_max += w_max_cur + wh_interval[0]
                li_li_idx.append(li_idx)
                li_li_xywh.append(li_xywh)
                y_cur = 0;  x_cur = w_max
                xywh = (x_cur, y_cur, w_cur, h_cur)
                li_idx = [idx]; li_xywh = [xywh]
                h_sum = h_cur + wh_interval[1]; w_max_cur = -1;
                y_cur = h_sum
            else:
                h_sum += h_cur + wh_interval[1]
                li_idx.append(idx)
                li_xywh.append(xywh)
                y_cur = h_sum
            w_max_cur = max(w_max_cur, w_cur)
        if 0 == w_max and 0 == h_max:
            h_max = h_sum
        li_li_idx.append(li_idx);   li_li_xywh.append(li_xywh)
        w_max += w_max_cur
        h_max -= wh_interval[1]
    #print('li_li_idx :', li_li_idx);    print('li_li_xywh :', li_li_xywh);   #exit(0)
    #print('h_max :', h_max);    print('w_max :', w_max);    #exit(0)
    return (w_max, h_max), li_li_idx, li_li_xywh

def is_this_empty_string(strin):
    return (strin in (None, '')) or (not strin.strip())

def concatenate_images(li_im, li_caption, is_horizontal, n_max_per_row_or_col, pxl_max_per_row_or_col, bgr_bg = (255, 255, 255), wh_interval = (0, 0), bgr_txt = (0, 0, 255)):
   
    if li_caption is not None:
        n_caption = len(li_caption)
    else:
        n_caption = 0
    li_hwc = [im.shape for im in li_im]
    wh_max, li_li_idx, li_li_xywh = compute_layout(li_hwc, is_horizontal, n_max_per_row_or_col, pxl_max_per_row_or_col, wh_interval)
    w_max, h_max = wh_max
    im = np.zeros((h_max, w_max, 3), np.uint8); 
    im[:] = bgr_bg
    #   for each row
    for li_idx, li_xywh in zip(li_li_idx, li_li_xywh): 
        #   for each col
        for idx, xywh in zip(li_idx, li_xywh):
        #   paste into the region.
            x_from = xywh[0];   y_from = xywh[1];   x_to = x_from + xywh[2];    y_to = y_from + xywh[3]
            if n_caption and idx < n_caption and False == is_this_empty_string(li_caption[idx]):
                im[y_from : y_to, x_from : x_to, :] = add_image_text(li_im[idx], li_caption[idx], bgr_txt)
            else:    
                im[y_from : y_to, x_from : x_to, :] = li_li_im[idx]
    #cv2.imwrite('temp.bmp', im); exit(0)
    return im    



class VideoReader(Dataset):
    def __init__(self, path, dir_img, transform=None):
        self.video = pims.PyAVVideoReader(path)
        self.rate = self.video.frame_rate
        self.transform = transform
        self.dir_img = dir_img
        if dir_img:
            print('dir_img :', dir_img); #   exit(0)
            self.dir_img = dir_img
            self.img_writer = ImageSequenceWriter(self.dir_img, 'png')
    @property
    def frame_rate(self):
        return self.rate
        
    def __len__(self):
        return len(self.video)
        
    def __getitem__(self, idx):
        frame = self.video[idx]
        #print('frame.shape 1 :', frame.shape)
        frame = Image.fromarray(np.asarray(frame))
        #print('frame.size 2 : {}, frame.mode : {}'.format(frame.size, frame.mode))
        if self.transform is not None:
            frame = self.transform(frame)
            #print('frame.shape 3 :', frame.shape)
        if self.dir_img:
            #print('frame.shape 4 :', frame.shape);  #exit(0)
            #frame = frame.unsqueeze(0)
            #print('frame.shape 5 :', frame.shape);  #exit(0)
            self.img_writer.write(frame.unsqueeze(0))
        return frame


class VideoWriter:
    def __init__(self, path, frame_rate, bit_rate=1000000):
        self.container = av.open(path, mode='w')
        self.stream = self.container.add_stream('h264', rate=f'{frame_rate:.4f}')
        self.stream.pix_fmt = 'yuv420p'
        self.stream.bit_rate = bit_rate
    
    def write(self, frames):
        # frames: [T, C, H, W]
        self.stream.width = frames.size(3)
        self.stream.height = frames.size(2)
        if frames.size(1) == 1:
            frames = frames.repeat(1, 3, 1, 1) # convert grayscale to RGB
        frames = frames.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
        for t in range(frames.shape[0]):
            frame = frames[t]
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            self.container.mux(self.stream.encode(frame))
                
    def close(self):
        self.container.mux(self.stream.encode())
        self.container.close()


class ImageSequenceReader(Dataset):
    def __init__(self, path, li_ext, shall_return_path, transform=None):
        #self.path = path
        #self.files = sorted(os.listdir(path))
        self.files = []
        for ext in li_ext:
            self.files += get_list_of_file_path_under_1st_with_3rd_extension(path, False, ext)
        assert self.files, 'There is NO image files whose extension is {} under {}'.format(li_ext, path)   
        #print('path :', path);
        #print('self.files :', self.files);  exit(0)
        self.shall_return_path = shall_return_path
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        '''
        t0 = os.path.join(self.path, self.files[idx])
        print('self.path : {}'.format(self.path));  #exit(0)
        print('self.files[idx] : {}'.format(self.files[idx]));  #exit(0)
        print('t0 :', t0);  exit(0)
        with Image.open(os.path.join(self.path, self.files[idx])) as img:
        '''
        with Image.open(self.files[idx]) as img:
            img.load()
        if self.transform is not None:
            '''
            #print('111');   exit(0)     #   111
            print('type(img) b4 : {}'.format(type(img)));       #   PIL.PngImagePlugin.PngImageFile
            img = self.transform(img)
            print('torch.max(img) : {}, torch.min(img) : {}'.format(torch.max(img), torch.min(img)))    #   torch.max(img) : 1.0, torch.min(img) : 0.0
            print('type(img) after : {}'.format(type(img)));    #   torch.Tensor    #exit(0)
            exit(0)
            '''
            #return {'img': self.transform(img), 'path': self.files[idx]} if self.shall_return_path else self.transform(img)
            return self.transform(img), self.files[idx] if self.shall_return_path else self.transform(img)
            #return self.transforms(img), img if self.shall_return_path else self.transforms(img)
        #print('222');   exit(0)         #   This is NOT reached.
        #return {'img': img, 'path': self.files[idx]} if self.shall_return_path else img
        return img, self.files[idx] if self.shall_return_path else img


class ImageSequenceWriter:
    def __init__(self, path, extension='jpg'):
        self.path = path
        self.extension = extension
        self.counter = 0
        rm_and_mkdir(path)
        #os.makedirs(path, exist_ok=True)
    
    def write(self, frames, li_id):
        # frames: [T, C, H, W]
        for t in range(frames.shape[0]):
            if li_id:
                aidi = li_id[t]
            else:
                aidi = str(self.counter).zfill(5)
            #to_pil_image(frames[t]).save(os.path.join(self.path, str(self.counter).zfill(5) + '.' + self.extension))
            to_pil_image(frames[t]).save(os.path.join(self.path, aidi + '.' + self.extension))
            self.counter += 1
            
    def close(self):
        pass
        
