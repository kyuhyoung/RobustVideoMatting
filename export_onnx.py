"""
python export_onnx.py \
    --model-variant mobilenetv3 \
    --checkpoint rvm_mobilenetv3.pth \
    --precision float32 \
    --opset 12 \
    --device cuda \
    --output model.onnx
    
Note:
    The device is only used for exporting. It has nothing to do with the final model.
    Float16 must be exported through cuda. Float32 can be exported through cpu.
"""

import argparse
import torch

from math import ceil, floor

from model import MattingNetwork, str2bool
#from model import MattingNetwork, round_i, str2bool

class Exporter:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.export()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
        parser.add_argument('--model-refiner', type=str, default='deep_guided_filter', choices=['deep_guided_filter', 'fast_guided_filter'])
        parser.add_argument('--precision', type=str, required=True, choices=['float16', 'float32'])
        parser.add_argument('--wh', type=str, required=True)
        parser.add_argument('--opset', type=int, required=True)
        parser.add_argument('--downsample-ratio', type=float, required=True)
        parser.add_argument('--device', type=str, required=True)
        parser.add_argument('--checkpoint', type=str, required=False)
        parser.add_argument('--output', type=str, required=True)
        parser.add_argument('--ceil_mode', type=str, required=True)
        self.args = parser.parse_args()
        
    def init_model(self):
        li_wh = self.args.wh.split('_');    self.w_img = int(li_wh[0]);  self.h_img = int(li_wh[1]);
        self.ceil_mode = str2bool(self.args.ceil_mode)
        self.precision = torch.float32 if self.args.precision == 'float32' else torch.float16
        self.model = MattingNetwork(self.ceil_mode, self.args.model_variant, self.args.model_refiner).eval().to(self.args.device, self.precision)
        if self.args.checkpoint is not None:
            #print('aaa');
            self.model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device), strict=False)
        #exit(0);
    def export(self):
        rec = [None] * 4
        '''
        if self.ceil_mode:
            w_0 = ceil((self.w_img * self.args.downsample_ratio - 2) / 2) + 1;
            w_1 = ceil((w_0 - 2) / 2) + 1;
            w_2 = ceil((w_1 - 2) / 2) + 1;
            w_3 = ceil((w_2 - 2) / 2) + 1;
            h_0 = ceil((self.h_img * self.args.downsample_ratio - 2) / 2) + 1;
            h_1 = ceil((h_0 - 2) / 2) + 1;
            h_2 = ceil((h_1 - 2) / 2) + 1;
            h_3 = ceil((h_2 - 2) / 2) + 1;
        else:
            w_0 = floor((self.w_img * self.args.downsample_ratio - 2) / 2 + 1);
            w_1 = floor((w_0 - 2) / 2 + 1);
            w_2 = floor((w_1 - 2) / 2 + 1);
            w_3 = floor((w_2 - 2) / 2 + 1);
            h_0 = floor((self.h_img * self.args.downsample_ratio - 2) / 2 + 1);
            h_1 = floor((h_0 - 2) / 2 + 1);
            h_2 = floor((h_1 - 2) / 2 + 1);
            h_3 = floor((h_2 - 2) / 2 + 1);
        '''
        w_0 = floor((self.w_img * self.args.downsample_ratio - 1 ) / 2 + 1);
        w_1 = floor((w_0 - 1) / 2 + 1);
        w_2 = floor((w_1 - 1) / 2 + 1);
        w_3 = floor((w_2 - 1) / 2 + 1);
        h_0 = floor((self.h_img * self.args.downsample_ratio - 1) / 2 + 1);
        h_1 = floor((h_0 - 1) / 2 + 1);
        h_2 = floor((h_1 - 1) / 2 + 1);
        h_3 = floor((h_2 - 1) / 2 + 1);
        if 'mobilenetv3' == self.args.model_variant:
            rec[0] = torch.zeros([1, 16, h_0, w_0]).to(self.args.device, self.precision)
            rec[1] = torch.zeros([1, 20, h_1, w_1]).to(self.args.device, self.precision)
            rec[2] = torch.zeros([1, 40, h_2, w_2]).to(self.args.device, self.precision)
            rec[3] = torch.zeros([1, 64, h_3, w_3]).to(self.args.device, self.precision)
        else:
            rec[0] = torch.zeros([1, 16, h_0, w_0]).to(self.args.device, self.precision)
            rec[1] = torch.zeros([1, 32, h_1, w_1]).to(self.args.device, self.precision)
            rec[2] = torch.zeros([1, 64, h_2, w_2]).to(self.args.device, self.precision)
            rec[3] = torch.zeros([1, 128, h_3, w_3]).to(self.args.device, self.precision)
        #rec[0] = torch.zeros([1, 16, 135, 240]).to(self.args.device, self.precision)
        #rec[1] = torch.zeros([1, 20, 68, 120]).to(self.args.device, self.precision)
        #rec[2] = torch.zeros([1, 40, 34, 60]).to(self.args.device, self.precision)
        #rec[3] = torch.zeros([1, 64, 17, 30]).to(self.args.device, self.precision)

        src = torch.randn(1, 3, self.h_img, self.w_img).to(self.args.device, self.precision)
        print('rec[0].shape :', rec[0].shape);  print('rec[1].shape :', rec[1].shape);
        print('rec[2].shape :', rec[2].shape);  print('rec[3].shape :', rec[3].shape);
        print('src.shape :', src.shape);
        #exit(0);
        downsample_ratio = torch.tensor([self.args.downsample_ratio]).to(self.args.device)
        '''
        dynamic_spatial = {0: 'batch_size', 2: 'height', 3: 'width'}
        dynamic_everything = {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'}
        '''
        torch.onnx.export(
            model = self.model,
            args = (src, *rec, downsample_ratio),
            f = self.args.output,
            export_params = True,
            #verbose = False,
            verbose = True,
            opset_version=self.args.opset,
            do_constant_folding=True,
            input_names=['src', 'r1i', 'r2i', 'r3i', 'r4i', 'downsample_ratio'],
            output_names=['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o'],
            #dynamic_axes={
            #    'src': dynamic_spatial,
            #    'fgr': dynamic_spatial,
            #    'pha': dynamic_spatial,
            #    'r1i': dynamic_everything,
            #    'r2i': dynamic_everything,
            #    'r3i': dynamic_everything,
            #    'r4i': dynamic_everything,
            #    'r1o': dynamic_spatial,
            #    'r2o': dynamic_spatial,
            #    'r3o': dynamic_spatial,
            #    'r4o': dynamic_spatial,
            #}
        )

if __name__ == '__main__':
    Exporter()
