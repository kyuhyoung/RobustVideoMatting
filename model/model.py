import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder, Projection, _interpolate_kevin
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner

#from .onnx_helper import CustomOnnxResizeByFactorOp
from .onnx_helper import CustomOnnxResizeToMatchSizeOp

def round_i(x):
    return int(round(x))
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')    
    

class MattingNetwork(nn.Module):
    def __init__(self,
                ceil_mode,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        if variant == 'mobilenetv3':
            #print('pretrained_backbone : {}'.format(pretrained_backbone));  exit(0)
            self.backbone = MobileNetV3LargeEncoder(ceil_mode, pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16], ceil_mode)
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16], ceil_mode)
            
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self, src, r1, r2, r3, r4,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        '''
        if torch.onnx.is_in_onnx_export():
            #print('aaa')
            src_sm = CustomOnnxResizeByFactorOp.apply(src, downsample_ratio)
        elif downsample_ratio != 1:
            #print('bbb')
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            #print('ccc')
            src_sm = src
        '''
        if downsample_ratio != 1:
            if torch.onnx.is_in_onnx_export():
                '''
                src_sm = CustomOnnxResizeByFactorOp.apply(src, downsample_ratio)
                '''
                #'''
                print('aaa')
                print('src.shape :', src.shape);    #exit(0)
                print('src.requires_grad :', src.requires_grad);    #exit(0)
                #print('torch.memory_format(src) :', torch.memory_format(src));    #exit(0)
                print('src :', src);    #exit(0)
                bchw = list(src.size())
                print('bchw :', bchw);  #exit(0)
                bb = bchw[0].item();   cc = bchw[1].item();   hh = round_i(bchw[2].item() * downsample_ratio.item());   ww = round_i(bchw[3].item() * downsample_ratio.item()); 
                #bb = int(src.shape[0]);  cc = int(src.shape[1]);  hh = round_i(float(src.shape[2]) * downsample_ratio);  ww = round_i(float(src.shape[3]) * downsample_ratio); 
                t0 = torch.zeros(bb, cc, hh, ww, dtype = src.dtype, layout = src.layout, device = src.device, requires_grad = False) 
                src_sm = CustomOnnxResizeToMatchSizeOp.apply(src, t0)
                #'''
                '''
                bchw = list(src.size())
                bb = bchw[0].item();   cc = bchw[1].item();   hh = round_i(bchw[2].item() * downsample_ratio.item());   ww = round_i(bchw[3].item() * downsample_ratio.item()); 
                hw = [hh, ww]
                #print('src.shape :', src.shape);
                src_sm = _interpolate_kevin(src, hw)
                #print('src_sm.shape :', src_sm.shape);    exit(0);
                '''
            else:
                #print('bbb')
                src_sm = self._interpolate(src, scale_factor=downsample_ratio)
            #exit(0)     
        else:
            src_sm = src


        #exit(0)     
        
        f1, f2, f3, f4 = self.backbone(src_sm)
        #print('f4.shape b4 : {}'.format(f4.shape))
        f4 = self.aspp(f4)
        #print('f4.shape : after {}'.format(f4.shape));  exit(0)
        '''
        print('f1.shape : {}, f2.shape : {}, f3.shape : {}, f4.shape : {}'.format(f1.shape, f2.shape, f3.shape, f4.shape));   #exit(0);
        print('r1.shape : {}, r2.shape : {}, r3.shape : {}, r4.shape : {}'.format(r1.shape, r2.shape, r3.shape, r4.shape));   exit(0);
        '''
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        
        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if torch.onnx.is_in_onnx_export() or downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.reshape(B, T, x.size(1), x.size(2), x.size(3))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
