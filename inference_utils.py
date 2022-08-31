#import av
#import pims
import os
import numpy as np
import torch
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
        self.stream = self.container.add_stream('h264', rate=round(frame_rate))
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
    def __init__(self, path, ext, transform=None):
        self.path = path
        #self.files = sorted(os.listdir(path))
        self.files = get_list_of_file_path_under_1st_with_3rd_extension(path, False, ext)
        #print('path :', path);
        #print('self.files :', self.files);  exit(0)
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        #'''
        t0 = os.path.join(self.path, self.files[idx])
        print('t0 :', t0);  #exit(0)
        #'''
        with Image.open(os.path.join(self.path, self.files[idx])) as img:
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
            return self.transform(img)
        #print('222');   exit(0)         #   This is NOT reached.
        return img


class ImageSequenceWriter:
    def __init__(self, path, extension='jpg'):
        self.path = path
        self.extension = extension
        self.counter = 0
        os.makedirs(path, exist_ok=True)
    
    def write(self, frames):
        # frames: [T, C, H, W]
        for t in range(frames.shape[0]):
            to_pil_image(frames[t]).save(os.path.join(
                self.path, str(self.counter).zfill(5) + '.' + self.extension))
            self.counter += 1
            
    def close(self):
        pass
        
