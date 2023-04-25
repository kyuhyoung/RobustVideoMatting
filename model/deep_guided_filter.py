import torch
from torch import nn
from torch.nn import functional as F

"""
Adopted from <https://github.com/wuhuikai/DeepGuidedFilter/>
"""

class DeepGuidedFilterRefiner(nn.Module):
    def __init__(self, hid_channels=16):
        super().__init__()
        self.box_filter = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False, groups=4)
        self.box_filter.weight.data[...] = 1 / 9
        self.conv = nn.Sequential(
            nn.Conv2d(4 * 2 + hid_channels, hid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(hid_channels, 4, kernel_size=1, bias=True)
        )
        
    def forward_single_frame(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        #print(f'fine_src.shape : {fine_src.shape}, base_src.shape : {base_src.shape}, base_fgr.shape : {base_fgr.shape}, base_pha.shape : {base_pha.shape}, base_hid.shape : {base_hid.shape}');    exit() 
        #   fine_src : 3, 2160, 3840    base_src : 3, 288, 512  base_fgr : 3, 288, 512  base_pha : 1, 288, 512  base_hid : 16, 288, 512
        fine_x = torch.cat([fine_src, fine_src.mean(1, keepdim=True)], dim=1)
        #print(f'fine_x.shape : {fine_x.shape}');    exit()  #   4, 2160, 3840 
        base_x = torch.cat([base_src, base_src.mean(1, keepdim=True)], dim=1)
        #print(f'base_x.shape : {base_x.shape}');    exit()  #   4, 288, 512
        base_y = torch.cat([base_fgr, base_pha], dim=1)     
        #print(f'base_y.shape : {base_y.shape}');    exit()  #   4, 288, 512
        
        mean_x = self.box_filter(base_x)
        #print(f'mean_x.shape : {mean_x.shape}');    #exit()  #   4, 288, 512
        mean_y = self.box_filter(base_y)
        #print(f'mean_y.shape : {mean_y.shape}');    #exit()  #   4, 288, 512
        cov_xy = self.box_filter(base_x * base_y) - mean_x * mean_y
        #print(f'cov_xy.shape : {cov_xy.shape}');    #exit()  #   4, 288, 512
        var_x  = self.box_filter(base_x * base_x) - mean_x * mean_x
        #print(f'var_x.shape : {var_x.shape}');    #exit()  #   4, 288, 512
        
        #t0 = torch.cat([cov_xy, var_x, base_hid], dim=1);   print(f't0.shape : {t0.shape}');    #   24, 288, 512
        A = self.conv(torch.cat([cov_xy, var_x, base_hid], dim=1))
        #print(f'A.shape : {A.shape}');    exit()#   4, 288, 512
        b = mean_y - A * mean_x                 #   4, 288, 512
        
        H, W = fine_src.shape[2:]               #   2160, 3840
        A = F.interpolate(A, (H, W), mode='bilinear', align_corners=False)
        b = F.interpolate(b, (H, W), mode='bilinear', align_corners=False)
        
        out = A * fine_x + b
        print(f'out.shape : {out.shape}');    exit()#   4, 288, 512
        fgr, pha = out.split([3, 1], dim=1)
        return fgr, pha
    
    def forward_time_series(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        B, T = fine_src.shape[:2]
        fgr, pha = self.forward_single_frame(
            fine_src.flatten(0, 1),
            base_src.flatten(0, 1),
            base_fgr.flatten(0, 1),
            base_pha.flatten(0, 1),
            base_hid.flatten(0, 1))
        fgr = fgr.unflatten(0, (B, T))
        pha = pha.unflatten(0, (B, T))
        return fgr, pha
    
    def forward(self, fine_src, base_src, base_fgr, base_pha, base_hid):
        if fine_src.ndim == 5:
            return self.forward_time_series(fine_src, base_src, base_fgr, base_pha, base_hid)
        else:
            return self.forward_single_frame(fine_src, base_src, base_fgr, base_pha, base_hid)
