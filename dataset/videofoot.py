
from torch.utils.data import Dataset

class VideoFootDataset(Dataset):
    def __init__(self,
                videofoot_dir,
                #size,
                seq_length,
                seq_sampler,
                transform=None):
        self.videofoot_dir = videofoot_dir
        self.videofoot_clips = sorted(os.listdir(videofoot_dir))
        self.videofoot_frames = [sorted(os.listdir(os.path.join(videofoot_dir, clip))) for clip in self.videofoot_clips]
            #print('self.videomatte_frames : {}'.format(self.videomatte_frames));  exit(0) # [['00001.jpg', '00002.jpg', ... '01999.jpg'], ... ['00001.jpg', '00002.jpg', ... '03231.jpg']]
        self.videomatte_idx = [(clip_idx, frame_idx) for clip_idx in range(len(self.videofoot_clips)) 
                               for frame_idx in range(0, len(self.videofoot_frames[clip_idx]), seq_length)]
        #print('self.videomatte_idx : {}'.format(self.videomatte_idx));  exit(0) # [(0, 0), (0, 15), ... , (0, 345), (1, 0), (1, 15), ... (1, 345), ... (474, 0), (474, 1), ... , (474, 435)]
        #self.size = size
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler
        self.transform = transform

    def __len__(self):
        return len(self.videofoot_idx)
    
    def __getitem__(self, idx):
        clip_idx, frame_idx = self.videofoot_idx
        li_rgb, li_heatmap = 
        if random.random() < 0.5:
            bgrs = self._get_random_image_background()
        else:
 
