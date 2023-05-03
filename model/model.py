import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder, Projection, ConvGRU
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner

class MattFootNetwork(nn.Module):
    def __init__(self, 
                matt_weight_path,
                variant: str = 'mobilenetv3',
                matt_ref: str = 'deep_guided_filter',
                pretrained_backbone: bool = False):
        super().__init__()
        self.net_matt = MattingNetwork(variant, matt_ref, pretrained_backbone)
        print(f'matt_weight_path : {matt_weight_path}');    #exit()
        self.net_matt.load_state_dict(torch.load(matt_weight_path))

        # freeze the weights of HumanSegmentationNet
        for param in self.net_matt.parameters():
            param.requires_grad = False

        if variant == 'mobilenetv3':
            self.net_foot = FootPosNetwork(128, 24) 
        else:
            #   TODO
            self.net_foot = FootPosNetwork(128, 24) 
             
    def forward(self,
                src: Tensor,
                r1_matt: Optional[Tensor] = None,
                r2_matt: Optional[Tensor] = None,
                r3_matt: Optional[Tensor] = None,
                r4_matt: Optional[Tensor] = None,
                r_foot: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        out_matt, r1_matt, r2_matt, r3_matt, r4_matt, in_foot = self.net_matt(src, r1_matt, r2_matt, r3_matt, r4_matt, downsample_ratio)                 
        out_foot = self.net_foot(in_foot, r_foot)
        return out_matt, r1_matt, r2_matt, r3_matt, r4_matt, out_foot


class FootPosNetwork(nn.Module):
    def __init__(self, n_in_aspp, n_mid_aspp):
        super().__init__()
        self.aspp1 = LRASPP(n_in_aspp, n_mid_aspp)
        self.gru = ConvGRU(n_mid_aspp // 2)
        #self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.aspp2 = LRASPP(n_mid_aspp, 1)


    def forward_single_frame(self, x):
        x = self.aspp1(x)
        x = self.gru(x)
        x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(x)
        x = self.aspp2(x)
        x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(x)
        return x
    
    def forward_time_series(self, x):
        B, T = s0.shape[:2]
        x = x.flatten(0, 1)
        x = self.forward_single_frame(x)
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)





class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        #print(f'pretrained_backbone : {pretrained_backbone}');  #   False exit() 
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])
            
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        #print(f'src.shape : {src.shape}');   # 1, 1, 3, 2160, 3840 #exit()    
        if downsample_ratio != 1:
            #print('a'); 
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            #print('b');
            src_sm = src
        #print(f'src.shape : {src.shape}, downsample_ratio : {downsample_ratio}, src_sm.shape : {src_sm.shape}');   exit()
        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        #print(f'hid.shape : {hid.shape}');    exit()  #   16, 288, 512 
        #print('segmentation_pass :', segmentation_pass);    return; 
        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            seg = seg.sigmoid()
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
