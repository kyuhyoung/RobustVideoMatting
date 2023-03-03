import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .onnx_helper import CustomOnnxCropToMatchSizeOp

#def _interpolate_kevin(x: Tensor, s: Tensor):
    #x = F.interpolate(x, s.shape[2:], mode='bilinear', align_corners=False)
    #return x
def _interpolate_kevin(x: Tensor, sz: torch.Size):
    print('x.shape b4 :', x.shape);
    print('type(sz) :', type(sz));# 
    print('sz :', sz);  #exit(0)
    #print('type(s.shape) :', type(s.shape));    exit(0);
    x = F.interpolate(x, sz, mode='bilinear', align_corners=False)
    print('x.shape after :', x.shape);  #exit(0)
    return x



class RecurrentDecoder(nn.Module):
    def __init__(self, feature_channels, decoder_channels, ceil_mode):
        super().__init__()
        #self.avgpool = AvgPool()
        self.avgpool = AvgPool(ceil_mode)
        self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2])
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

    def forward(self, s0, f1, f2, f3, f4, r1, r2, r3, r4):
        s1, s2, s3 = self.avgpool(s0);
        #print('s0.shape : {}, s1.shape : {}, s2.shape : {}, s3.shape : {}'.format(s0.shape, s1.shape, s2.shape, s3.shape));  exit(0)
        #print(f'f4.shape : {f4.shape}, r4.shape : {r4.shape}');  exit(0)
        x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(x4, f3, s3, r3)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        x1, r1 = self.decode1(x2, f1, s1, r1)
        x0 = self.decode0(x1, s0)
        return x0, r1, r2, r3, r4
    

class AvgPool(nn.Module):
    def __init__(self, ceil_mode):
        super().__init__()
        #self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        if ceil_mode:
            self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        else:     
            self.avgpool = nn.AvgPool2d(3, stride = 2, padding = 1, count_include_pad=False)
            #iself.avgpool = nn.MaxPool2d(3, i2, stride = 2)
        
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
        
    def forward(self, x, r):
        #print('\nx.shape b4 : {}, r.shape b4 : {}\n'.format(x.shape, r.shape))
        a, b = x.split(self.channels // 2, dim=-3)
        #print('\na.shape : {}, b.shape b4 : {}\n'.format(a.shape, b.shape))
        b, r = self.gru(b, r)
        #print('\nb.shape after : {}, r.shape after : {}\n'.format(b.shape, r.shape))
        x = torch.cat([a, b], dim=-3)
        #print('\nx.shape after : {}\n'.format(x.shape))
        return x, r

    
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.gru = ConvGRU(out_channels // 2)

    def forward_single_frame(self, x, f, s, r):
        #x = self.upsample(x)
        #x = F.interpolate(x, s.shape[2:], mode='bilinear', align_corners=False)
        print('s.shape :', s.shape)
        print('x.shape b4 :', x.shape)
        x = _interpolate_kevin(x, s.shape[2:])
        print('x.shape after :', x.shape);   #exit(0)
        # if not torch.onnx.is_in_onnx_export():
            # x = x[:, :, :s.size(2), :s.size(3)]
        # else:
            # x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=1)
        return x, r
    
    def forward_time_series(self, x, f, s, r):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        #x = self.upsample(x)
        x = _interpolate_kevin(x, s.shape[2:])
        #x = F.interpolate(x, s.shape[2:], mode='bilinear', align_corners=False)
        # if not torch.onnx.is_in_onnx_export():
            # x = x[:, :, :H, :W]
        # else:
            # x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=2)
        return x, r
    
    def forward(self, x, f, s, r):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r)
        else:
            return self.forward_single_frame(x, f, s, r)


class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        #self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        #x = self.upsample(x)
        x = _interpolate_kevin(x, s.shape[2:])
        #x = F.interpolate(x, s.shape[2:], mode='bilinear', align_corners=False)
        # if not torch.onnx.is_in_onnx_export():
            # x = x[:, :, :s.size(1), :s.size(2)]
        # else:
            # x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        #x = self.upsample(x)
        x = _interpolate_kevin(x, s.shape[2:])
        #x = F.interpolate(x, s.shape[2:], mode='bilinear', align_corners=False)
        # x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
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
        print(f'channels : {channels}, kernel_size : {kernel_size}');    #exit(0)
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )
        
    def forward_single_frame(self, x, h):
        #print(f'x.shape : {x.shape}, h.shape : {h.shape}'); exit(0)
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
        
    def forward(self, x, h):
        '''
        h_b4 = h.clone().detach()
        h = h.expand_as(x)
        if not torch.equal(h_b4, h):
            print('h_b4 :', h_b4);
            print('h :', h);    exit(0) #   This does NOT happen, which means h_b4 and h are alway equeal and expand as is not necessary.
        '''    
        #h = h.expand_as(x)
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
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
    
