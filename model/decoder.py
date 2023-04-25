import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional

class RecurrentDecoder(nn.Module):
    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2])
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

    def forward(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor],
                r3: Optional[Tensor], r4: Optional[Tensor]):
        
        #   r1 : None    r2 : None     r3 : None      r4 : None
        #   f1 : 1, 16, 144, 256    f2 : 1, 24, 72, 128     f3 : 1, 40, 36, 64      f4 : 1, 128, 18, 32
        s1, s2, s3 = self.avgpool(s0)
        #print(f's1.shape : {s1.shape}, s2.shape : {s2.shape}, s3.shape : {s3.shape}');  
        #   s1 : 1, 1, 3, 144, 256      s2 : 1, 1, 3, 72, 128       s3 : 1, 1, 3, 36, 64    #   exit()
        x4, r4 = self.decode4(f4, r4)
        #   x4 : 128, 18, 32
        x3, r3 = self.decode3(x4, f3, s3, r3)
        #print(f'x3.shape : {x3.shape}');    exit()  #   80, 36, 64 
        x2, r2 = self.decode2(x3, f2, s2, r2)
        #print(f'x2.shape : {x2.shape}');    exit()  #   40, 72, 128 
        x1, r1 = self.decode1(x2, f1, s1, r1)
        #print(f'x1.shape : {x1.shape}');    exit()  #   32, 144, 256 
        x0 = self.decode0(x1, s0)
        #print(f'x0.shape : {x0.shape}');    exit()  #   16, 288, 512 
        return x0, r1, r2, r3, r4
    

class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        
    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        return s1, s2, s3
    
    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        s1 = s1.unflatten(0, (B, T))
        s2 = s2.unflatten(0, (B, T))
        s3 = s3.unflatten(0, (B, T))
        return s1, s2, s3
    
    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)
        
    def forward(self, x, r: Optional[Tensor]):
        a, b = x.split(self.channels // 2, dim=-3)
        #print(f'x.shape : {x.shape}, a.shape : {a.shape}, b.shape : {b.shape}');    exit()
        #   x : 128, 18, 32     a : 64, 18, 32       b : 64, 18, 32  
        b, r = self.gru(b, r)
        #print(f'b.shape : {b.shape}');    # 64, 18, 32  #exit()
        x = torch.cat([a, b], dim=-3)
        #print(f'x.shape : {x.shape}');    128, 18, 32  #exit()
        return x, r

    
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.gru = ConvGRU(out_channels // 2)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor]):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=1)
        return x, r
    
    def forward_time_series(self, x, f, s, r: Optional[Tensor]):
        B, T, _, H, W = s.shape     #   1, 1, 3, 36, 64
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        #shall_debug = 40 == x.shape[1]
        #if shall_debug:
        #    print(f'x.shape b4 : {x.shape}');   exit()       #   1, 128, 18 32   //  1, 80, 36, 64 //  1, 40, 72, 128
        x = self.upsample(x)
        #print(f'x.shape after : {x.shape}');       #   1, 128, 36, 64  //  1, 80, 72, 128          
        x = x[:, :, :H, :W]                     #   128, 36, 64
        #if shall_debug:
        #    print(f'x.shape after 2 : {x.shape}');  #   1, 128, 36, 64  //  1, 80, 72, 128  //  1, 40, 144, 256    
        x = torch.cat([x, f, s], dim=1)
        #if shall_debug:
        #    print(f'x.shape after 3 : {x.shape}');  #   1, 171, 36, 64  //  1, 107, 72, 128 //  1, 59, 144, 256
        x = self.conv(x)
        #if shall_debug:
        #    print(f'x.shape after 4 : {x.shape}');  # 1, 80, 36, 64 //  1, 40, 72, 128  //  1, 32, 144, 256  
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=2)
        #print(f'x.shape after 5 : {x.shape}');    exit()# 80, 36, 64
        return x, r
    
    def forward(self, x, f, s, r: Optional[Tensor]):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r)
        else:
            return self.forward_single_frame(x, f, s, r)


class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        #print(f'x.shape : {x.shape}');     exit() #   32, 144, 256
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        #print(f'x.shape : {x.shape}');     exit() #   32, 288, 512
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        #print(f'x.shape : {x.shape}');     exit() #   35, 288, 512
        x = self.conv(x)
        #print(f'x.shape : {x.shape}');     exit() #   16, 288, 512
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)


class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )
        
    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h
    
    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h
        
    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward_single_frame(self, x):
        return self.conv(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        #print(f'x.shape : {x.shape}');  exit()  #   16, 288, 512
        #t0 = self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        #print(f't0.shape : {t0.shape}');  exit()  #   4, 288, 512
        #t1, t2 = ha = t0.split([3, 1], dim=-3)
        #print(f't1.shape : {t1.shape}, t2.shape : {t2.shape}');  exit()  #   t1 : 3, 288, 512   t2 : 1, 288, 512
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
    
