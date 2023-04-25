import os
import torch
from torch import nn
from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidualConfig
from torchvision.transforms.functional import normalize

class MobileNetV3LargeEncoder(MobileNetV3):
    def __init__(self, pretrained: bool = False):
        super().__init__(
            inverted_residual_setting=[
                InvertedResidualConfig( 16, 3,  16,  16, False, "RE", 1, 1, 1),
                InvertedResidualConfig( 16, 3,  64,  24, False, "RE", 2, 1, 1),  # C1
                InvertedResidualConfig( 24, 3,  72,  24, False, "RE", 1, 1, 1),
                InvertedResidualConfig( 24, 5,  72,  40,  True, "RE", 2, 1, 1),  # C2
                InvertedResidualConfig( 40, 5, 120,  40,  True, "RE", 1, 1, 1),
                InvertedResidualConfig( 40, 5, 120,  40,  True, "RE", 1, 1, 1),
                InvertedResidualConfig( 40, 3, 240,  80, False, "HS", 2, 1, 1),  # C3
                InvertedResidualConfig( 80, 3, 200,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 184,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 184,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 480, 112,  True, "HS", 1, 1, 1),
                InvertedResidualConfig(112, 3, 672, 112,  True, "HS", 1, 1, 1),
                InvertedResidualConfig(112, 5, 672, 160,  True, "HS", 2, 2, 1),  # C4
                InvertedResidualConfig(160, 5, 960, 160,  True, "HS", 1, 2, 1),
                InvertedResidualConfig(160, 5, 960, 160,  True, "HS", 1, 2, 1),
            ],
            last_channel=1280
        )
        
        if pretrained:
            self.load_state_dict(torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'))

        del self.avgpool
        del self.classifier
        
    def forward_single_frame(self, x):
        #print(f'x.shape : {x.shape}')   #   1, 3, 288, 512  #exit()
        #print('torch.max(x) b4 : {}, torch.min(x) b4 : {}'.format(torch.max(x), torch.min(x)))          #   torch.max(x) : 1.0, torch.min(x) : 0.0
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #print('torch.max(x) after : {}, torch.min(x) after : {}'.format(torch.max(x), torch.min(x)))    #   torch.max(x) : 2.64, torch.min(x) : -2.12
        #exit(0);
        x = self.features[0](x)
        x = self.features[1](x)
        f1 = x
        x = self.features[2](x)
        x = self.features[3](x)
        f2 = x
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        f3 = x
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        f4 = x
        #print(f'f1.shape : {f1.shape}, f2.shape : {f2.shape}, f3.shape : {f3.shape}, f4.shape : {f4.shape}');   exit()
        #   f1 : 1, 16, 144, 256    f2 : 1, 24, 72, 128     f3 : 1, 40, 36, 64      f4 : 1, 960, 18, 32
        return [f1, f2, f3, f4]
    
    def forward_time_series(self, x):
        #print(f'x.shape : {x.shape}');  #   1, 1, 3, 288, 512   #exit()
        B, T = x.shape[:2]
        #t0 = x.flatten(0, 1);   print(f't0.shape : {t0.shape}');    #   1, 3, 288, 512  #exit()
        features = self.forward_single_frame(x.flatten(0, 1))
        #   features[0] : 1, 16, 144, 256    features[1] : 1, 24, 72, 128     features[2] : 1, 40, 36, 64      features[3] : 1, 960, 18, 32
        features = [f.unflatten(0, (B, T)) for f in features]
        #   features[0] : 1, 1, 16, 144, 256    features[1] : 1, 1, 24, 72, 128     features[2] : 1, 1, 1, 40, 36, 64      features[3] : 1, 1, 960, 18, 32
        return features

    def forward(self, x):
        #print(f'x.ndim : {x.ndim}');    #   5   #exit()
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
