"""
# First update `train_config.py` to set paths to your dataset locations.

# You may want to change `--num-workers` according to your machine's memory.
# The default num-workers=8 may cause dataloader to exit unexpectedly when
# machine is out of memory.

# Stage 1
python train.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 15 \
    --learning-rate-backbone 0.0001 \
    --learning-rate-aspp 0.0002 \
    --learning-rate-decoder 0.0002 \
    --learning-rate-refiner 0 \
    --checkpoint-dir checkpoint/stage1 \
    --log-dir log/stage1 \
    --epoch-start 0 \
    --epoch-end 20

# Stage 2
python train.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 50 \
    --learning-rate-backbone 0.00005 \
    --learning-rate-aspp 0.0001 \
    --learning-rate-decoder 0.0001 \
    --learning-rate-refiner 0 \
    --checkpoint checkpoint/stage1/epoch-19.pth \
    --checkpoint-dir checkpoint/stage2 \
    --log-dir log/stage2 \
    --epoch-start 20 \
    --epoch-end 22
    
# Stage 3
python train.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 40 \
    --seq-length-hr 6 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00001 \
    --learning-rate-refiner 0.0002 \
    --checkpoint checkpoint/stage2/epoch-21.pth \
    --checkpoint-dir checkpoint/stage3 \
    --log-dir log/stage3 \
    --epoch-start 22 \
    --epoch-end 23

# Stage 4
python train.py \
    --model-variant mobilenetv3 \
    --dataset imagematte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 40 \
    --seq-length-hr 6 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00005 \
    --learning-rate-refiner 0.0002 \
    --checkpoint checkpoint/stage3/epoch-22.pth \
    --checkpoint-dir checkpoint/stage4 \
    --log-dir log/stage4 \
    --epoch-start 23 \
    --epoch-end 28
"""


import argparse
import torch
import random
import os
import numpy as np
import cv2
from torch import nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms.functional import center_crop
from tqdm import tqdm
from inference_utils import rm_and_mkdir, tensor_0_1_to_ndarray_0_255, concatenate_images
from utils import get_exact_file_name_from_path


from dataset.videomatte import (
    VideoMatteDataset,
    VideoMatteTrainAugmentation,
    VideoMatteValidAugmentation,
)

#'''
from dataset.imagematte import (
    ImageMatteDataset,
    ImageMatteAugmentation
)
#'''

from dataset.coco import (
    CocoPanopticDataset,
    CocoPanopticTrainAugmentation
)
from dataset.figaro1k import (
    Figaro1kDataset
)
#'''
from dataset.spd import (
    SuperviselyPersonDataset
)
#'''

from dataset.celebamask import (
    CelebAMaskDataset
)

#'''
from dataset.youtubevis import (
    YouTubeVISDataset,
    YouTubeVISAugmentation
)
#'''

from dataset.augmentation import (
    TrainFrameSampler,
    ValidFrameSampler
)

from model import MattingNetwork
from train_config import DATA_PATHS
from train_loss import matting_loss, segmentation_loss


class Trainer:
    def __init__(self, rank, world_size):
        self.parse_args()
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        self.init_writer()
        self.train()
        self.cleanup()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Model
        parser.add_argument('--model-variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
        # Matting dataset
        parser.add_argument('--dataset', type=str, required=True, choices=['videomatte', 'imagematte'])
        # Learning rate
        parser.add_argument('--learning-rate-backbone', type=float, required=True)
        parser.add_argument('--learning-rate-aspp', type=float, required=True)
        parser.add_argument('--learning-rate-decoder', type=float, required=True)
        parser.add_argument('--learning-rate-refiner', type=float, required=True)
        # Training setting
        parser.add_argument('--train-hr', action='store_true')
        parser.add_argument('--is_hair', action='store_true')
        parser.add_argument('--resolution-lr', type=int, default=512)
        parser.add_argument('--resolution-hr', type=int, default=2048)
        parser.add_argument('--seq-length-lr', type=int, required=True)
        parser.add_argument('--seq-length-hr', type=int, default=6)
        parser.add_argument('--downsample-ratio', type=float, default=0.25)
        parser.add_argument('--batch-size-per-gpu', type=int, default=1)
        #parser.add_argument('--batch-size-per-gpu', type=int, default=3)
        parser.add_argument('--num-workers', type=int, default=8)
        parser.add_argument('--epoch-start', type=int, default=0)
        parser.add_argument('--epoch-end', type=int, default=16)
        # Tensorboard logging
        parser.add_argument('--log-dir', type=str, required=True)
        parser.add_argument('--log-train-loss-interval', type=int, default=20)
        parser.add_argument('--log-train-images-interval', type=int, default=500)
        # Checkpoint loading and saving
        parser.add_argument('--checkpoint', type=str)
        parser.add_argument('--checkpoint-dir', type=str, required=True)
        #parser.add_argument('--checkpoint-save-interval', type=int, default=500)
        parser.add_argument('--checkpoint-save-interval', type=int, default=5000)
        # Distributed
        parser.add_argument('--distributed-addr', type=str, default='localhost')
        parser.add_argument('--distributed-port', type=str, default='12355')
        # Debugging
        parser.add_argument('--disable-progress-bar', action='store_true')
        parser.add_argument('--disable-validation', action='store_true')
        parser.add_argument('--disable-mixed-precision', action='store_true')
        self.args = parser.parse_args()
        
    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.log('Initializing distributed')
        os.environ['MASTER_ADDR'] = self.args.distributed_addr
        os.environ['MASTER_PORT'] = self.args.distributed_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    def init_datasets(self):
        self.log('Initializing matting datasets')
        size_hr = (self.args.resolution_hr, self.args.resolution_hr)
        size_lr = (self.args.resolution_lr, self.args.resolution_lr)
        
        # Matting datasets:
        if self.args.dataset == 'videomatte':
            self.dataset_lr_train = VideoMatteDataset(
                is_hair = self.args.is_hair,
                #videomatte_dir=DATA_PATHS['videomatte']['train'],
                videomatte_dir = DATA_PATHS['videomatte']['train_hair'] if self.args.is_hair else DATA_PATHS['videomatte']['train_portrait'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=DATA_PATHS['background_videos']['train'],
                size=self.args.resolution_lr,
                seq_length=self.args.seq_length_lr,
                #seq_sampler=TrainFrameSampler(),
                seq_sampler = TrainFrameSampler(speed = [1]) if self.args.is_hair else TrainFrameSampler(),
                transform=VideoMatteTrainAugmentation(size_lr))
            if self.args.train_hr:
                self.dataset_hr_train = VideoMatteDataset(
                    is_hair = self.args.is_hair,
                    #videomatte_dir=DATA_PATHS['videomatte']['train'],
                    videomatte_dir = DATA_PATHS['videomatte']['train_hair'] if self.args.is_hair else DATA_PATHS['videomatte']['train_portrait'],
                    background_image_dir=DATA_PATHS['background_images']['train'],
                    background_video_dir=DATA_PATHS['background_videos']['train'],
                    size=self.args.resolution_hr,
                    seq_length=self.args.seq_length_hr,
                    #seq_sampler=TrainFrameSampler(),
                    seq_sampler = TrainFrameSampler(speed = [1]) if self.args.is_hair else TrainFrameSampler(),
                    transform=VideoMatteTrainAugmentation(size_hr))
            self.dataset_valid = VideoMatteDataset(
                is_hair = self.args.is_hair,
                #videomatte_dir=DATA_PATHS['videomatte']['valid'],
                videomatte_dir = DATA_PATHS['videomatte']['valid_hair'] if self.args.is_hair else DATA_PATHS['videomatte']['valid_portrait'],
                background_image_dir=DATA_PATHS['background_images']['valid'],
                background_video_dir=DATA_PATHS['background_videos']['valid'],
                size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
                seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
                seq_sampler=ValidFrameSampler(),
                transform=VideoMatteValidAugmentation(size_hr if self.args.train_hr else size_lr))
        else:
            self.dataset_lr_train = ImageMatteDataset(
                imagematte_dir=DATA_PATHS['imagematte']['train'],
                background_image_dir=DATA_PATHS['background_images']['train'],
                background_video_dir=DATA_PATHS['background_videos']['train'],
                size=self.args.resolution_lr,
                seq_length=self.args.seq_length_lr,
                seq_sampler=TrainFrameSampler(),
                transform=ImageMatteAugmentation(size_lr))
            if self.args.train_hr:
                self.dataset_hr_train = ImageMatteDataset(
                    imagematte_dir=DATA_PATHS['imagematte']['train'],
                    background_image_dir=DATA_PATHS['background_images']['train'],
                    background_video_dir=DATA_PATHS['background_videos']['train'],
                    size=self.args.resolution_hr,
                    seq_length=self.args.seq_length_hr,
                    seq_sampler=TrainFrameSampler(),
                    transform=ImageMatteAugmentation(size_hr))
            self.dataset_valid = ImageMatteDataset(
                imagematte_dir=DATA_PATHS['imagematte']['valid'],
                background_image_dir=DATA_PATHS['background_images']['valid'],
                background_video_dir=DATA_PATHS['background_videos']['valid'],
                size=self.args.resolution_hr if self.args.train_hr else self.args.resolution_lr,
                seq_length=self.args.seq_length_hr if self.args.train_hr else self.args.seq_length_lr,
                seq_sampler=ValidFrameSampler(),
                transform=ImageMatteAugmentation(size_hr if self.args.train_hr else size_lr))
            
        # Matting dataloaders:
        self.datasampler_lr_train = DistributedSampler(
            dataset=self.dataset_lr_train,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)
        self.dataloader_lr_train = DataLoader(
            dataset=self.dataset_lr_train,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_lr_train,
            pin_memory=True)
        if self.args.train_hr:
            self.datasampler_hr_train = DistributedSampler(
                dataset=self.dataset_hr_train,
                rank=self.rank,
                num_replicas=self.world_size,
                shuffle=True)
            self.dataloader_hr_train = DataLoader(
                dataset=self.dataset_hr_train,
                batch_size=self.args.batch_size_per_gpu,
                num_workers=self.args.num_workers,
                sampler=self.datasampler_hr_train,
                pin_memory=True)
        self.dataloader_valid = DataLoader(
            dataset=self.dataset_valid,
            batch_size=self.args.batch_size_per_gpu,
            num_workers=self.args.num_workers,
            pin_memory=True)
        
        # Segementation datasets
        self.log('Initializing image segmentation datasets')
        if self.args.is_hair:
            self.dataset_seg_image = ConcatDataset([
                Figaro1kDataset(
                    imgdir=DATA_PATHS['figaro1k']['imgdir'],
                    segdir=DATA_PATHS['figaro1k']['segdir'],
                    transform=CocoPanopticTrainAugmentation(size_lr)),
                CelebAMaskDataset(
                    root_dir = DATA_PATHS['celeb_a_mask_hq']['rootdir'],
                    is_train = True,
                    #segdir=DATA_PATHS['celeb_a_mask_hq']['segdir'],
                    transforms=CocoPanopticTrainAugmentation(size_lr))
            ])
        else:
            self.dataset_seg_image = ConcatDataset([
                CocoPanopticDataset(
                    imgdir=DATA_PATHS['coco_panoptic']['imgdir'],
                    anndir=DATA_PATHS['coco_panoptic']['anndir'],
                    annfile=DATA_PATHS['coco_panoptic']['annfile'],
                    transform=CocoPanopticTrainAugmentation(size_lr)),
                SuperviselyPersonDataset(
                    imgdir=DATA_PATHS['spd']['imgdir'],
                    segdir=DATA_PATHS['spd']['segdir'],
                    transform=CocoPanopticTrainAugmentation(size_lr))
            ])
        self.datasampler_seg_image = DistributedSampler(
            dataset=self.dataset_seg_image,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)
        self.dataloader_seg_image = DataLoader(
            dataset=self.dataset_seg_image,
            batch_size=self.args.batch_size_per_gpu * self.args.seq_length_lr,
            num_workers=self.args.num_workers,
            sampler=self.datasampler_seg_image,
            pin_memory=True)
       
        if False == self.args.is_hair: 
            self.log('Initializing video segmentation datasets')
            self.dataset_seg_video = YouTubeVISDataset(
                videodir=DATA_PATHS['youtubevis']['videodir'],
                annfile=DATA_PATHS['youtubevis']['annfile'],
                size=self.args.resolution_lr,
                seq_length=self.args.seq_length_lr,
                seq_sampler=TrainFrameSampler(speed=[1]),
                transform=YouTubeVISAugmentation(size_lr))
            self.datasampler_seg_video = DistributedSampler(
                dataset=self.dataset_seg_video,
                rank=self.rank,
                num_replicas=self.world_size,
                shuffle=True)
            self.dataloader_seg_video = DataLoader(
                dataset=self.dataset_seg_video,
                batch_size=self.args.batch_size_per_gpu,
                num_workers=self.args.num_workers,
                sampler=self.datasampler_seg_video,
                pin_memory=True)
        
    def init_model(self):
        self.log('Initializing model')
        self.model = MattingNetwork(self.args.model_variant, pretrained_backbone=True).to(self.rank)
        
        if self.args.checkpoint:
            self.log(f'Restoring from checkpoint: {self.args.checkpoint}')
            self.log(self.model.load_state_dict(
                torch.load(self.args.checkpoint, map_location=f'cuda:{self.rank}')))
            
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model_ddp = DDP(self.model, device_ids=[self.rank], broadcast_buffers=False, find_unused_parameters=True)
        self.optimizer = Adam([
            {'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate_backbone},
            {'params': self.model.aspp.parameters(), 'lr': self.args.learning_rate_aspp},
            {'params': self.model.decoder.parameters(), 'lr': self.args.learning_rate_decoder},
            {'params': self.model.project_mat.parameters(), 'lr': self.args.learning_rate_decoder},
            {'params': self.model.project_seg.parameters(), 'lr': self.args.learning_rate_decoder},
            {'params': self.model.refiner.parameters(), 'lr': self.args.learning_rate_refiner},
        ])
        self.scaler = GradScaler()
        
    def init_writer(self):
        if self.rank == 0:
            self.log('Initializing writer')
            self.writer = SummaryWriter(self.args.log_dir)
        
    def train(self):
        
        rm_and_mkdir(self.args.checkpoint_dir)
        stage = get_exact_file_name_from_path(self.args.checkpoint_dir)
        dir_check = 'output/train_check/{}'.format(stage)
        #print('dir_check : {}'.format(dir_check)); exit(0)
        rm_and_mkdir(dir_check)

        tu_fgr_pha_bgr_li_fn = None
        for epoch in range(self.args.epoch_start, self.args.epoch_end):
            self.epoch = epoch
            self.step = epoch * len(self.dataloader_lr_train)
            '''
            if not self.args.disable_validation:
                self.validate()
            '''
            self.log(f'Training epoch: {epoch}')
            i_batch = 0
            if self.args.is_hair:
                for true_fgr, true_pha_body, true_pha_hair, true_bgr, li_fn_fgr in tqdm(self.dataloader_lr_train, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                    # Low resolution pass
                    self.train_mat((true_fgr, true_pha_body, true_pha_hair, true_bgr), downsample_ratio=1, tag='lr')
                    # High resolution pass
                    if self.args.train_hr:
                        true_fgr, true_pha_body, true_pha_hair, true_bgr, li_fn_fgr = self.load_next_mat_hr_sample()
                        self.train_mat((true_fgr, true_pha_body, true_pha_hair, true_bgr), downsample_ratio=self.args.downsample_ratio, tag='hr')
                    # Segmentation pass
                    #if self.step % 2 == 0:
                    if False:
                        true_img, true_seg = self.load_next_seg_video_sample()
                        self.train_seg(true_img, true_seg, log_label='seg_video')
                    else:
                        true_img, true_seg = self.load_next_seg_image_sample()
                        self.train_seg(true_img.unsqueeze(1), true_seg.unsqueeze(1), log_label='seg_image')
 
                    if None == tu_fgr_pha_bgr_li_fn:
                        tu_fgr_pha_bgr_li_fn = (true_fgr, true_pha_body, true_pha_hair, true_bgr, li_fn_fgr)
                        
                    #print('self.step : {}, self.args.checkpoint_save_interval : {}'.format(self.step, self.args.checkpoint_save_interval))
                    if self.step % self.args.checkpoint_save_interval == 0:
                        #self.save()
                        loss_val = self.validate()
                        fn_save = 'ep_{:02d}_ib_{:05d}_val_loss_{}.pth'.format(self.epoch, i_batch, loss_val) 
                        self.save(fn_save)
                        self.check_progress(dir_check, self.epoch, i_batch, tu_fgr_pha_bgr_li_fn)
                        self.log('Train check is done at self.step : {} and model is saved at {}'.format(self.step, fn_save))
                   
                   
                    '''
                    if i_batch % self.args.checkpoint_save_interval == 0:
                        self.check_progress(dir_check, epoch, i_batch, tu_fgr_pha_bgr_li_fn)
                        print('Train check is done at self.step : {}'.format(self.step))
                    '''
                    self.step += 1
                    i_batch += 1


            else:
                #for true_fgr, true_pha, true_bgr in tqdm(self.dataloader_lr_train, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                for true_fgr, true_pha, true_bgr, li_fn_fgr in tqdm(self.dataloader_lr_train, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                    # Low resolution pass
                    #self.train_mat(true_fgr, true_pha, true_bgr, downsample_ratio=1, tag='lr')
                    self.train_mat((true_fgr, true_pha, true_bgr), downsample_ratio=1, tag='lr')

                    # High resolution pass
                    if self.args.train_hr:
                        true_fgr, true_pha, true_bgr = self.load_next_mat_hr_sample()
                        #self.train_mat(true_fgr, true_pha, true_bgr, downsample_ratio=self.args.downsample_ratio, tag='hr')
                        self.train_mat((true_fgr, true_pha, true_bgr), downsample_ratio=self.args.downsample_ratio, tag='hr')
                    # Segmentation pass
                    if self.step % 2 == 0:
                        true_img, true_seg = self.load_next_seg_video_sample()
                        self.train_seg(true_img, true_seg, log_label='seg_video')
                    else:
                        true_img, true_seg = self.load_next_seg_image_sample()
                        self.train_seg(true_img.unsqueeze(1), true_seg.unsqueeze(1), log_label='seg_image')
 
                    if None == tu_fgr_pha_bgr_li_fn:
                        tu_fgr_pha_bgr_li_fn = (true_fgr, true_pha, true_bgr, li_fn_fgr)
                        
                    #print('self.step : {}, self.args.checkpoint_save_interval : {}'.format(self.step, self.args.checkpoint_save_interval))
                    if self.step % self.args.checkpoint_save_interval == 0:
                        self.save()
                        #self.save(fn_save)
                        self.check_progress(dir_check, self.epoch, i_batch, tu_fgr_pha_bgr_li_fn)
                        self.log('Train check is done at self.step : {} and mocel is saved.'.format(self.step))
                    
                   
                   
                    self.step += 1
                    i_batch += 1

    #def train_mat(self, true_fgr, true_pha, true_bgr, downsample_ratio, tag):
    def train_mat(self, tu_true, downsample_ratio, tag):
        n_true = len(tu_true)
        is_hair = 4 == n_true
        if is_hair:
            true_fgr, true_pha_body, true_pha_hair, true_bgr = tu_true
            true_fgr = true_fgr.to(self.rank, non_blocking=True)
            true_pha_body = true_pha_body.to(self.rank, non_blocking=True)
            true_pha_hair = true_pha_hair.to(self.rank, non_blocking=True)
            true_bgr = true_bgr.to(self.rank, non_blocking=True)
            true_fgr, true_pha_body, true_pha_hair, true_bgr = self.random_crop(true_fgr, true_pha_body, true_pha_hair, true_bgr)
            #true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
            true_src = true_fgr * true_pha_body + true_bgr * (1 - true_pha_body)
            
            with autocast(enabled=not self.args.disable_mixed_precision):
                #pred_fgr, pred_pha = self.model_ddp(true_src, downsample_ratio=downsample_ratio)[:2]
                pred_fgr, pred_pha_hair = self.model_ddp(true_src, downsample_ratio=downsample_ratio)[:2]
                #loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)
                loss = matting_loss(pred_fgr, pred_pha_hair, true_fgr, true_pha_hair)

        else:
            true_fgr, true_pha, true_bgr = tu_true
            true_fgr = true_fgr.to(self.rank, non_blocking=True)
            true_pha = true_pha.to(self.rank, non_blocking=True)
            true_bgr = true_bgr.to(self.rank, non_blocking=True)
            true_fgr, true_pha, true_bgr = self.random_crop(true_fgr, true_pha, true_bgr)
            true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
            
            with autocast(enabled=not self.args.disable_mixed_precision):
                pred_fgr, pred_pha = self.model_ddp(true_src, downsample_ratio=downsample_ratio)[:2]
                loss = matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)

        self.scaler.scale(loss['total']).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        if self.rank == 0 and self.step % self.args.log_train_loss_interval == 0:
            for loss_name, loss_value in loss.items():
                self.writer.add_scalar(f'train_{tag}_{loss_name}', loss_value, self.step)
            
        if self.rank == 0 and self.step % self.args.log_train_images_interval == 0:
            if is_hair:
                self.writer.add_image(f'train_{tag}_pred_fgr', make_grid(pred_fgr.flatten(0, 1), nrow=pred_fgr.size(1)), self.step)
                self.writer.add_image(f'train_{tag}_pred_pha_hair', make_grid(pred_pha_hair.flatten(0, 1), nrow=pred_pha_hair.size(1)), self.step)
                self.writer.add_image(f'train_{tag}_true_fgr', make_grid(true_fgr.flatten(0, 1), nrow=true_fgr.size(1)), self.step)
                self.writer.add_image(f'train_{tag}_true_pha_hair', make_grid(true_pha_hair.flatten(0, 1), nrow=true_pha_hair.size(1)), self.step)
                self.writer.add_image(f'train_{tag}_true_src', make_grid(true_src.flatten(0, 1), nrow=true_src.size(1)), self.step)

            else:
                self.writer.add_image(f'train_{tag}_pred_fgr', make_grid(pred_fgr.flatten(0, 1), nrow=pred_fgr.size(1)), self.step)
                self.writer.add_image(f'train_{tag}_pred_pha', make_grid(pred_pha.flatten(0, 1), nrow=pred_pha.size(1)), self.step)
                self.writer.add_image(f'train_{tag}_true_fgr', make_grid(true_fgr.flatten(0, 1), nrow=true_fgr.size(1)), self.step)
                self.writer.add_image(f'train_{tag}_true_pha', make_grid(true_pha.flatten(0, 1), nrow=true_pha.size(1)), self.step)
                self.writer.add_image(f'train_{tag}_true_src', make_grid(true_src.flatten(0, 1), nrow=true_src.size(1)), self.step)
                
    def train_seg(self, true_img, true_seg, log_label):
        true_img = true_img.to(self.rank, non_blocking=True)
        true_seg = true_seg.to(self.rank, non_blocking=True)
        
        true_img, true_seg = self.random_crop(true_img, true_seg)
        
        with autocast(enabled=not self.args.disable_mixed_precision):
            pred_seg = self.model_ddp(true_img, segmentation_pass=True)[0]
            loss = segmentation_loss(pred_seg, true_seg)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        if self.rank == 0 and (self.step - self.step % 2) % self.args.log_train_loss_interval == 0:
            self.writer.add_scalar(f'{log_label}_loss', loss, self.step)
        
        if self.rank == 0 and (self.step - self.step % 2) % self.args.log_train_images_interval == 0:
            self.writer.add_image(f'{log_label}_pred_seg', make_grid(pred_seg.flatten(0, 1).float().sigmoid(), nrow=self.args.seq_length_lr), self.step)
            self.writer.add_image(f'{log_label}_true_seg', make_grid(true_seg.flatten(0, 1), nrow=self.args.seq_length_lr), self.step)
            self.writer.add_image(f'{log_label}_true_img', make_grid(true_img.flatten(0, 1), nrow=self.args.seq_length_lr), self.step)
    
    def load_next_mat_hr_sample(self):
        try:
            sample = next(self.dataiterator_mat_hr)
        except:
            self.datasampler_hr_train.set_epoch(self.datasampler_hr_train.epoch + 1)
            self.dataiterator_mat_hr = iter(self.dataloader_hr_train)
            sample = next(self.dataiterator_mat_hr)
        return sample
    
    def load_next_seg_video_sample(self):
        try:
            sample = next(self.dataiterator_seg_video)
        except:
            self.datasampler_seg_video.set_epoch(self.datasampler_seg_video.epoch + 1)
            self.dataiterator_seg_video = iter(self.dataloader_seg_video)
            sample = next(self.dataiterator_seg_video)
        return sample
    
    def load_next_seg_image_sample(self):
        try:
            sample = next(self.dataiterator_seg_image)
        except:
            self.datasampler_seg_image.set_epoch(self.datasampler_seg_image.epoch + 1)
            self.dataiterator_seg_image = iter(self.dataloader_seg_image)
            sample = next(self.dataiterator_seg_image)
        return sample
    
    def validate(self):
        #print('self.rank : {}'.format(self.rank));    #exit(0)   #   0 
        avg_loss = None
        if self.rank == 0:
            #self.log(f'Validating at the start of epoch: {self.epoch}')
            self.log('Validating ...')
            self.model_ddp.eval()
            total_loss, total_count = 0, 0
            with torch.no_grad():
                with autocast(enabled=not self.args.disable_mixed_precision):
                    #for tu_fgr_pha_bgr in tqdm(self.dataloader_valid, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                    if self.args.is_hair:
                        for true_fgr, true_pha_body, true_pha_hair, true_bgr, li_fn_fgr in tqdm(self.dataloader_valid, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                            #print('li_fn_fgr : {}'.format(li_fn_fgr));  exit(0)
                            true_fgr = true_fgr.to(self.rank, non_blocking=True)
                            true_pha_body = true_pha_body.to(self.rank, non_blocking=True)
                            true_pha_hair = true_pha_hair.to(self.rank, non_blocking=True)
                            true_bgr = true_bgr.to(self.rank, non_blocking=True)
                            true_src = true_fgr * true_pha_body + true_bgr * (1 - true_pha_body)
                            batch_size = true_src.size(0)
                            pred_fgr, pred_pha_hair = self.model(true_src)[:2]
                            total_loss += matting_loss(pred_fgr, pred_pha_hair, true_fgr, true_pha_hair)['total'].item() * batch_size
                            total_count += batch_size
                    else:
                        for true_fgr, true_pha, true_bgr, li_fn_fgr in tqdm(self.dataloader_valid, disable=self.args.disable_progress_bar, dynamic_ncols=True):
                            #print('type(di_li_fn_fgr) : {}'.format(type(di_li_fn_fgr)));  #exit()  #   list
                            print('type(li_fn_fgr) : {}'.format(type(li_fn_fgr)));  #exit()  #   list
                            print('len(li_fn_fgr) : {}'.format(len(li_fn_fgr)));  #exit()    #   15
                            true_fgr = true_fgr.to(self.rank, non_blocking=True)
                            true_pha = true_pha.to(self.rank, non_blocking=True)
                            true_bgr = true_bgr.to(self.rank, non_blocking=True)
                            true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)

                            print('type(true_src) : {}'.format(type(true_src)));  #exit(0);  #   Tensor  
                            print('true_src.shape : {}'.format(true_src.shape));  #exit(0);  #   [1, 15, 3, 512, 512]  
                            #exit(0)
                            batch_size = true_src.size(0)
                            pred_fgr, pred_pha = self.model(true_src)[:2]
                            total_loss += matting_loss(pred_fgr, pred_pha, true_fgr, true_pha)['total'].item() * batch_size
                            total_count += batch_size
            avg_loss = total_loss / total_count
            self.log(f'Validation set average loss: {avg_loss}')
            self.writer.add_scalar('valid_loss', avg_loss, self.step)
            self.model_ddp.train()
        dist.barrier()
        return avg_loss

    def random_crop(self, *imgs):
        h, w = imgs[0].shape[-2:]
        w = random.choice(range(w // 2, w))
        h = random.choice(range(h // 2, h))
        results = []
        for img in imgs:
            B, T = img.shape[:2]
            img = img.flatten(0, 1)
            img = F.interpolate(img, (max(h, w), max(h, w)), mode='bilinear', align_corners=False)
            img = center_crop(img, (h, w))
            img = img.reshape(B, T, *img.shape[1:])
            results.append(img)
        return results
   
    def check_progress(self, dir_check, i_epoch, i_batch, tu_fgr_pha_bgr_li_fn):
        self.model_ddp.eval()
        with torch.no_grad():
            #solid_rgb = None
            #true_fgr, true_pha, true_bgr = tu_fgr_pha_bgr
            n_true = len(tu_fgr_pha_bgr_li_fn)
            is_hair = 5 == n_true
            if is_hair:
                true_fgr, true_pha, true_pha_hair, true_bgr, li_fn_fgr = tu_fgr_pha_bgr_li_fn
                true_pha_hair = true_pha_hair.to(self.rank, non_blocking=True)
            else:
                true_fgr, true_pha, true_bgr, li_fn_fgr = tu_fgr_pha_bgr_li_fn
            true_fgr = true_fgr.to(self.rank, non_blocking=True)
            true_pha = true_pha.to(self.rank, non_blocking=True)
            true_bgr = true_bgr.to(self.rank, non_blocking=True)
            true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
            batch_size = true_src.size(0)
            pred_fgr, pred_pha = self.model(true_src)[:2]
            #print('batch_size : {}'.format(batch_size))
            #print('type(pred_fgr) : {}, pred_fgr.shape : {}'.format(type(pred_fgr), pred_fgr.shape))
            #print('type(pred_pha) : {}, pred_pha.shape : {}'.format(type(pred_pha), pred_pha.shape))
            #print('torch.min(pred_fgr) : {}, torch.max(pred_fgr) : {}'.format(torch.min(pred_fgr), torch.max(pred_fgr)))
            #print('torch.min(pred_pha) : {}, torch.max(pred_pha) : {}'.format(torch.min(pred_pha), torch.max(pred_pha)))
            n_frm = pred_fgr.shape[1]; height = pred_fgr.shape[3]; width = pred_fgr.shape[4]
            rgb = (1.0, 0.0, 0.0)
            t0 = np.full((batch_size, n_frm, height, width, len(rgb)), rgb)
            t1 = torch.from_numpy(t0)
            #print('type(t1) : {}, t1.shape : {}'.format(type(t1), t1.shape))
            solid_rgb = t1.permute(0, 1, 4, 2, 3)
            #print('type(solid_rgb) : {}, solid_rgb.shape : {}'.format(type(solid_rgb), solid_rgb.shape))
            solid_rgb = solid_rgb.to(self.rank, non_blocking=True)
            pred_comp = pred_fgr * pred_pha + solid_rgb * (1 - pred_pha)
            if is_hair:
                true_comp = true_fgr * true_pha_hair + solid_rgb * (1 - true_pha_hair)
            else:
                true_comp = true_fgr * true_pha + solid_rgb * (1 - true_pha)
            for i_sam in range(batch_size):
                for i_frm in range(n_frm):
                    #print('len(li_fn_fgr) : {}'.format(len(li_fn_fgr)));
                    #print('li_fn_fgr : {}'.format(li_fn_fgr));   #exit(0)
                    #print('i_frm {} / {}, li_fn_fgr[i_frm] : {}'.format(i_frm, n_frm, li_fn_fgr[i_frm]));   exit(0)
                    fn_check = os.path.join(dir_check, '{:02d}_{:05d}_{:01d}_{:02d}'.format(i_epoch, i_batch, i_sam, i_frm) + '.png')
                    #t2 = pred_src[i_sam, i_frm]
                    #print('type(t2) : {}, t2.shape : {}'.format(type(t2), t2.shape))
                    bgr_pred_comp = cv2.cvtColor(tensor_0_1_to_ndarray_0_255(pred_comp[i_sam, i_frm]), cv2.COLOR_RGB2BGR)
                    bgr_true_src = cv2.cvtColor(tensor_0_1_to_ndarray_0_255(true_src[i_sam, i_frm]), cv2.COLOR_RGB2BGR)
                    bgr_pred_fgr = cv2.cvtColor(tensor_0_1_to_ndarray_0_255(pred_fgr[i_sam, i_frm]), cv2.COLOR_RGB2BGR)
                    bgr_true_comp = cv2.cvtColor(tensor_0_1_to_ndarray_0_255(true_comp[i_sam, i_frm]), cv2.COLOR_RGB2BGR)
                    bgr_concat = concatenate_images([bgr_true_src, bgr_pred_fgr, bgr_true_comp, bgr_pred_comp], ['true_src', 'pred_fgr', 'true_comp', 'pred_comp'], True, 2, 100000, (0, 0, 0), (0, 0), (255, 0, 0)) 
                    '''
                    t3 = t2.permute(1, 2, 0)
                    t4 = t3.cpu().detach().numpy()
                    t5 = 255 * t4
                    t6 = np.uint8(t5)
                    '''
                    #t7 = cv2.cvtColor(t6, cv2.COLOR_BGR2RGB)
                    #t7 = cv2.cvtColor(t3, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(fn_check, bgr_concat)
        self.model_ddp.train()
            
            
    #def save(self):
    def save(self, fn = None):
        if self.rank == 0:
            #os.makedirs(self.args.checkpoint_dir, exist_ok=True)
            #torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint_dir, f'epoch-{self.epoch}.pth'))
            if fn is None:
                fn = f'epoch-{self.epoch}.pth'
            torch.save(self.model.state_dict(), os.path.join(self.args.checkpoint_dir, fn))
            #self.log('Model saved')
        dist.barrier()
        
    def cleanup(self):
        dist.destroy_process_group()
        
    def log(self, msg):
        print(f'[GPU{self.rank}] {msg}')
            
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(
        Trainer,
        nprocs=world_size,
        args=(world_size,),
        join=True)
