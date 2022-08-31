"""
Expected directory format:

VideoMatte Train/Valid:
    ├──fgr/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
    ├── pha/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
        
ImageMatte Train/Valid:
    ├── fgr/
      ├── sample1.jpg
      ├── sample2.jpg
    ├── pha/
      ├── sample1.jpg
      ├── sample2.jpg

Background Image Train/Valid
    ├── sample1.png
    ├── sample2.png

Background Video Train/Valid
    ├── 0000/
      ├── 0000.jpg/
      ├── 0001.jpg/

"""


DATA_PATHS = {
    
    'videomatte': {
        #'train': '../matting-data/VideoMatte240K_JPEG_SD/train',
        #'valid': '../matting-data/VideoMatte240K_JPEG_SD/valid',
        'train': '/data/VideoMatte240K/VideoMatte240K_JPEG_SD/train',
        'valid': '/data/VideoMatte240K/VideoMatte240K_JPEG_SD/valid',
    },
    'imagematte': {
        'train': '../matting-data/ImageMatte/train',
        'valid': '../matting-data/ImageMatte/valid',
    },
    'background_images': {
        #'train': '../matting-data/Backgrounds/train',
        #'valid': '../matting-data/Backgrounds/valid',
        'train': '/data/background_matting_v2/Backgrounds_Validation/Backgrounds/train',
        'valid': '/data/background_matting_v2/Backgrounds_Validation/Backgrounds/valid',
    },
    'background_videos': {
        #'train': '../matting-data/BackgroundVideos/train',
        #'valid': '../matting-data/BackgroundVideos/valid',
        'train': '/data/dvm/BackgroundVideosTrain/train',
        'valid': '/data/dvm/BackgroundVideosTrain/valid',
    },
    
    
    'coco_panoptic': {
        #'imgdir': '../matting-data/coco/train2017/',
        'imgdir': '/data/coco/train2017/',
        #'anndir': '../matting-data/coco/panoptic_train2017/',
        'anndir': '/data/coco/panoptic_annotations_trainval2017/annotations/panoptic_train2017',
        #'annfile': '../matting-data/coco/annotations/panoptic_train2017.json',
        'annfile': '/data/coco/panoptic_annotations_trainval2017/annotations/panoptic_train2017.json',
    },
    'spd': {
        #'imgdir': '../matting-data/SuperviselyPersonDataset/img',
        'imgdir': '/data/SuperviselyPersonDataset/img',
        #'segdir': '../matting-data/SuperviselyPersonDataset/seg',
        'segdir': '/data/SuperviselyPersonDataset/seg',
    },
    'youtubevis': {
        #'videodir': '../matting-data/YouTubeVIS/train/JPEGImages',
        'videodir': '/data/YouTubeVIS/2021/train/JPEGImages',
        #'annfile': '../matting-data/YouTubeVIS/train/instances.json',
        'annfile': '/data/YouTubeVIS/2021/train/instances.json',
    }
    
}
