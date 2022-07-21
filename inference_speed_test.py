"""
python inference_speed_test.py \
    --model-variant mobilenetv3 \
    --resolution 1920 1080 \
    --downsample-ratio 0.25 \
    --precision float32
"""
import time
import argparse
import torch
from tqdm import tqdm

from model.model import MattingNetwork

torch.backends.cudnn.benchmark = True

class InferenceSpeedTest:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.loop()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-variant', type=str, required=True)
        parser.add_argument('--resolution', type=int, required=True, nargs=2)
        parser.add_argument('--downsample-ratio', type=float, required=True)
        parser.add_argument('--precision', type=str, default='float32')
        parser.add_argument('--disable-refiner', action='store_true')
        parser.add_argument('--ckpt', type=str, default = None)
        self.args = parser.parse_args()
        
    #def init_model(self):
    def init_model(self):
        self.device = 'cuda'
        self.precision = {'float32': torch.float32, 'float16': torch.float16}[self.args.precision]
        self.model = MattingNetwork(self.args.model_variant)
        self.model = self.model.to(device=self.device, dtype=self.precision).eval()
        if self.args.ckpt:
            self.model.load_state_dict(torch.load(self.args.ckpt, map_location=self.device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
    
    def loop(self):
        w, h = self.args.resolution
        src = torch.randn((1, 3, h, w), device=self.device, dtype=self.precision)
        with torch.no_grad():
            rec = None, None, None, None
            n_iter = 1000
            li_sec_total = []
            #for _ in tqdm(range(1000)):
            for idx in tqdm(range(n_iter)):
                start_total = time.time()
                fgr, pha, *rec = self.model(src, *rec, self.args.downsample_ratio)
                torch.cuda.synchronize()
                sec_total = time.time() - start_total
                #downsample_ratio         0.25  0.1
                #if idx >= 0:   #   fps : 109   34
                #if idx >= 10:  #   fps : 207   42
                if idx >= 20:  #   fps : 213    42 
                #if idx >= 40:   #   fps : 214   42
                    li_sec_total.append(sec_total)
            #avg_fps_total = float(n_iter) / float(sec_total)
            avg_fps_total = float(len(li_sec_total)) / float(sum(li_sec_total))
            print('avg_fps_total : {}'.format(avg_fps_total))
if __name__ == '__main__':
    InferenceSpeedTest()
