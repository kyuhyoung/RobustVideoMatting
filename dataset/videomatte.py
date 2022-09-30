import os
import random
from torch.utils.data import Dataset
from PIL import Image
import json

from .augmentation import MotionAugmentation

def get_exact_file_name_from_path(str_path):
    return os.path.splitext(os.path.basename(str_path))[0]

def is_this_existing_file(path_file):
    return os.path.exists(path_file) and os.path.isfile(path_file)

                    
def get_fn_img_from_json(path_json):
    fn_img = None
    #print('path_json : {}'.format(path_json))
    with open(path_json, 'r', encoding='utf-8') as phile:
        json_data = json.load(phile)
        fn_img = json_data['filename'] 
    assert fn_img, 'Can NOT find filename in the json file : {}'.format(path_json)
    return fn_img

class VideoMatteDataset(Dataset):
    def __init__(self,
                is_hair,
                 videomatte_dir,
                 background_image_dir,
                 background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform=None):
        self.is_hair = is_hair
        self.background_image_dir = background_image_dir
        self.background_image_files = os.listdir(background_image_dir)
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]
        
        self.videomatte_dir = videomatte_dir
        if self.is_hair:
            self.videomatte_clips = []
            self.videomatte_frames = []
            li_hair_style = sorted(os.listdir(videomatte_dir))
            for hair_style in li_hair_style:
                li_id = sorted(os.listdir(os.path.join(videomatte_dir, hair_style)))
                for aidi in li_id:
                    self.videomatte_clips.append('{}/{}'.format(hair_style, aidi))
                    path_dir_img = os.path.join(videomatte_dir, hair_style, aidi)
                    li_path_json = [os.path.join(path_dir_img, fn) for fn in sorted(os.listdir(path_dir_img)) if fn.lower().endswith('json')]
                    li_fn_img = sorted([get_fn_img_from_json(path_json) for path_json in li_path_json])
                    #li_fn_img = [fn for fn in os.listdir(path_dir_img) if (fn.lower().endswith('jpg') or fn.lower().endswith('jpeg'))]   
                    self.videomatte_frames.append(li_fn_img)
            #print('self.videomatte_clips : {}'.format(self.videomatte_clips));  exit(0) #   ['0001/4308.CP5634', '0001/8903.JP9493', ... '0200/3240.HU234907']
            #print('self.videomatte_frames : {}'.format(self.videomatte_frames));  exit(0) # [['00001.jpg', '00002.jpg', ... '01999.jpg'], ... ['00001.jpg', '00002.jpg', ... '03231.jpg']]
        else:
            self.videomatte_clips = sorted(os.listdir(os.path.join(videomatte_dir, 'fgr')))
            #print('self.videomatte_clips : {}'.format(self.videomatte_clips));  exit(0) #   ['0001', '0002', ... '0200']
            self.videomatte_frames = [sorted(os.listdir(os.path.join(videomatte_dir, 'fgr', clip))) 
                                  for clip in self.videomatte_clips]
            #print('self.videomatte_frames : {}'.format(self.videomatte_frames));  exit(0) # [['00001.jpg', '00002.jpg', ... '01999.jpg'], ... ['00001.jpg', '00002.jpg', ... '03231.jpg']]
        self.videomatte_idx = [(clip_idx, frame_idx) 
                               for clip_idx in range(len(self.videomatte_clips)) 
                               for frame_idx in range(0, len(self.videomatte_frames[clip_idx]), seq_length)]
        
        #print('self.videomatte_idx : {}'.format(self.videomatte_idx));  exit(0) # [(0, 0), (0, 15), ... , (0, 345), (1, 0), (1, 15), ... (1, 345), ... (474, 0), (474, 1), ... , (474, 435)]
        self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

    def __len__(self):
        return len(self.videomatte_idx)
    
    def __getitem__(self, idx):
        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
            bgrs = self._get_random_video_background()
       
        if self.is_hair:
            fgrs_body, phas_body, phas_hair, li_fn_fgr = self._get_videomatte(idx)
            if self.transform is not None:
                #return self.transform((fgrs_body, phas_body, phas_hair, bgrs))
                fgrs_body, phas_body, phas_hair, bgrs = self.transform((fgrs_body, phas_body, phas_hair, bgrs))
            return fgrs_body, phas_body, phas_hair, bgrs, li_fn_fgr
        else:
            fgrs, phas, li_fn_fgr = self._get_videomatte(idx)
            if self.transform is not None:
                '''
                print('type(fgrs) : {}'.format(type(fgrs)));  #exit(0);  #   list  
                print('len(fgrs) : {}'.format(len(fgrs)));  #exit(0);  #   15  
                print('type(fgrs[0]) : {}'.format(type(fgrs[0])));  #exit(0);  #   PIL.Image.Image  
                print('fgrs[0].size : {}, fgrs[0].mode : {}'.format(fgrs[0].size, fgrs[0].mode));  # (768, 432), RGB 
                li_fgr, li_pha, li_bgr = self.transform((fgrs, phas, bgrs))
                #print('type(li_fgr) : {}'.format(type(li_fgr)));  exit(0);  #   Tensor  
                print('li_fgr.shape : {}'.format(li_fgr.shape));  #exit(0);  #   [15, 3, 512, 512]  
                '''
                #return self.transform((fgrs, phas, bgrs))
                fgrs, phas, bgrs = self.transform((fgrs, phas, bgrs))
            return fgrs, phas, bgrs, li_fn_fgr
    
    def _get_random_image_background(self):
        with Image.open(os.path.join(self.background_image_dir, random.choice(self.background_image_files))) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        bgrs = [bgr] * self.seq_length
        return bgrs
    
    def _get_random_video_background(self):
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)
        return bgrs
    
    def _get_videomatte(self, idx):
        clip_idx, frame_idx = self.videomatte_idx[idx]
        clip = self.videomatte_clips[clip_idx]
        frame_count = len(self.videomatte_frames[clip_idx])
        if self.is_hair:
            fgrs_body, phas_body, phas_hair = [], [], []
        else:
            fgrs, phas = [], []
        li_fn_fgr = []      
        for i in self.seq_sampler(self.seq_length):
            if self.is_hair:
                #frame = self.videomatte_frames[clip_idx][(frame_idx + i) % frame_count]
                t0 = self.videomatte_frames[clip_idx][(frame_idx + i) % frame_count]
                id_img = get_exact_file_name_from_path(t0)
                #frame_fgr = '{}.jpg'.format(t1)
                #frame_pha = '{}.png'.format(t1)
                li_path_jpg_jpeg = []
                li_path_jpg_jpeg.append(os.path.join(self.videomatte_dir, clip, id_img + '.jpg'))
                li_path_jpg_jpeg.append(os.path.join(self.videomatte_dir, clip, id_img + '.JPG'))
                li_path_jpg_jpeg.append(os.path.join(self.videomatte_dir, clip, id_img + '.jpeg'))
                li_path_jpg_jpeg.append(os.path.join(self.videomatte_dir, clip, id_img + '.JPEG'))
                path_fgr_body = None
                for path_jpg_jpeg in li_path_jpg_jpeg:
                    if is_this_existing_file(path_jpg_jpeg):
                        path_fgr_body = path_jpg_jpeg
                        break
                assert path_fgr_body, 'There is no jpg or jpeg file for {}'.format(os.path.join(self.videomatte_dir, clip, id_img))
                path_pha_body = os.path.join(self.videomatte_dir, clip, 'rvm_alpha', id_img + '.png')
                path_pha_hair = os.path.join(self.videomatte_dir, clip, 'hair_seg_map', id_img + '.png')
                #with Image.open(os.path.join(self.videomatte_dir, clip, 'rvm_comp', frame)) as fgr_body, \
                with Image.open(path_fgr_body) as fgr_body, Image.open(path_pha_body) as pha_body, Image.open(path_pha_hair) as pha_hair:
                    if (fgr_body.size != pha_body.size) or (fgr_body.size != pha_hair.size) or (pha_body.size != pha_hair.size):
                        print('fgr_body.size : {}, pha_body.size : {}, pha_hair.size : {} at {}'.format(fgr_body.size, pha_body.size, pha_hair.size, path_fgr_body));  exit();
                        
                    fgr_body = self._downsample_if_needed(fgr_body.convert('RGB'))
                    pha_body = self._downsample_if_needed(pha_body.convert('L'))
                    pha_hair = self._downsample_if_needed(pha_hair.convert('L'))
                fgrs_body.append(fgr_body)
                phas_body.append(pha_body)
                phas_hair.append(pha_hair)
                li_fn_fgr.append(path_fgr_body)
            else:
                frame = self.videomatte_frames[clip_idx][(frame_idx + i) % frame_count]
                with Image.open(os.path.join(self.videomatte_dir, 'fgr', clip, frame)) as fgr, \
                 Image.open(os.path.join(self.videomatte_dir, 'pha', clip, frame)) as pha:
                    fgr = self._downsample_if_needed(fgr.convert('RGB'))
                    pha = self._downsample_if_needed(pha.convert('L'))
                fgrs.append(fgr)
                phas.append(pha)
                li_fn_fgr.append(os.path.join(self.videomatte_dir, 'fgr', clip, frame))
        #di_li_fn_fgr = {'li_fn_fgr' : li_fn_fgr}
        if self.is_hair:
            #return fgrs_body, phas_body, phas_hair, di_li_fn_fgr
            return fgrs_body, phas_body, phas_hair, li_fn_fgr
        else:
            #return fgrs, phas, di_li_fn_fgr
            return fgrs, phas, li_fn_fgr
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

class VideoMatteTrainAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.3,
            prob_bgr_affine=0.3,
            prob_noise=0.1,
            prob_color_jitter=0.3,
            prob_grayscale=0.02,
            prob_sharpness=0.1,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )

class VideoMatteValidAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0,
            prob_bgr_affine=0,
            prob_noise=0,
            prob_color_jitter=0,
            prob_grayscale=0,
            prob_sharpness=0,
            prob_blur=0,
            prob_hflip=0,
            prob_pause=0,
        )
