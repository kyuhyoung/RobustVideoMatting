from torch import nn
from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidualConfig
from torch.hub import load_state_dict_from_url
from torchvision.transforms.functional import normalize
from inspect import getfile as gf, currentframe as cf

class MobileNetV3LargeEncoder(MobileNetV3):
    def __init__(self, ceil_mode, pretrained: bool = False):
        #print('aaa');   exit(0)
        #'''    
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
        #'''
        '''
        super().__init__(
            inverted_residual_setting=[
                InvertedResidualConfig( 16, 3,  16,  16, False, "RE", 1, 1, 1),
                InvertedResidualConfig( 16, 3,  64,  24, False, "RE", 2, 1, 1),  # C1
                InvertedResidualConfig( 24, 3,  72,  24, False, "RE", 1, 1, 1),
                InvertedResidualConfig( 24, 5,  72,  40,  True if ceil_mode else False, "RE", 2, 1, 1),  # C2
                InvertedResidualConfig( 40, 5, 120,  40,  True if ceil_mode else False, "RE", 1, 1, 1),
                InvertedResidualConfig( 40, 5, 120,  40,  True if ceil_mode else False, "RE", 1, 1, 1),
                InvertedResidualConfig( 40, 3, 240,  80, False, "HS", 2, 1, 1),  # C3
                InvertedResidualConfig( 80, 3, 200,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 184,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 184,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 480, 112,  True if ceil_mode else False, "HS", 1, 1, 1),
                InvertedResidualConfig(112, 3, 672, 112,  True if ceil_mode else False, "HS", 1, 1, 1),
                InvertedResidualConfig(112, 5, 672, 160,  True if ceil_mode else False, "HS", 2, 2, 1),  # C4
                InvertedResidualConfig(160, 5, 960, 160,  True if ceil_mode else False, "HS", 1, 2, 1),
                InvertedResidualConfig(160, 5, 960, 160,  True if ceil_mode else False, "HS", 1, 2, 1),
            ],
            last_channel=1280
        )
        '''


        if pretrained:
            self.load_state_dict(load_state_dict_from_url(
                'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'))

        del self.avgpool
        del self.classifier
        
    def forward_single_frame(self, x):
        #print('x.shape 1 {}'.format(x.shape)) 
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #print('x.shape 2 {}'.format(x.shape)) 
        x = self.features[0](x)
        #print('x.shape 3 {}'.format(x.shape)) 
        x = self.features[1](x)
        #print('x.shape 4 {}'.format(x.shape)) 
        f1 = x
        x = self.features[2](x)
        #print('x.shape 5 {}'.format(x.shape)) 
        x = self.features[3](x)
        #print('x.shape 6 {}'.format(x.shape)) 
        f2 = x
        x = self.features[4](x)
        #print('x.shape 7 {}'.format(x.shape)) 
        x = self.features[5](x)
        #print('x.shape 8 {}'.format(x.shape)) 
        x = self.features[6](x)
        #print('x.shape 9 {}'.format(x.shape)) 
        f3 = x
        x = self.features[7](x)
        #print('x.shape 10 {}'.format(x.shape)) 
        x = self.features[8](x)
        #print('x.shape 11 {}'.format(x.shape)) 
        x = self.features[9](x)
        #print('x.shape 12 {}'.format(x.shape)) 
        x = self.features[10](x)
        #print('x.shape 13 {}'.format(x.shape)) 
        x = self.features[11](x)
        #print('x.shape 14 {}'.format(x.shape)) 
        x = self.features[12](x)
        #print('x.shape 15 {}'.format(x.shape)) 
        x = self.features[13](x)
        #print('x.shape 16 {}'.format(x.shape)) 
        x = self.features[14](x)
        #print('x.shape 17 {}'.format(x.shape)) 
        x = self.features[15](x)
        #print('x.shape 18 {}'.format(x.shape)) 
        x = self.features[16](x)
        #print('x.shape 19 {}'.format(x.shape)) 
        #exit(0)
        f4 = x
        return [f1, f2, f3, f4]
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]
        return features

    def forward(self, x):
        print(f'x.ndim : {x.ndim} at {gf(cf())} {cf().f_lineno}')
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
