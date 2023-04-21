import torch
import os
import time
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm
from utils import get_exact_file_name_from_path
from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter, get_list_of_file_path_under_1st_with_3rd_extension
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import shutil 


def image_to_tensor(image: "np.ndarray", keepdim: bool = True) -> torch.Tensor:
    """Convert a numpy image to a PyTorch 4d tensor image.
    Args:
        image: image of the form :math:`(H, W, C)`, :math:`(H, W)` or
            :math:`(B, H, W, C)`.
        keepdim: If ``False`` unsqueeze the input image to match the shape
            :math:`(B, H, W, C)`.
    Returns:
        tensor of the form :math:`(B, C, H, W)` if keepdim is ``False``,
            :math:`(C, H, W)` otherwise.
    Example:
        >>> img = np.ones((3, 3))
        >>> image_to_tensor(img).shape
        torch.Size([1, 3, 3])
        >>> img = np.ones((4, 4, 1))
        >>> image_to_tensor(img).shape
        torch.Size([1, 4, 4])
        >>> img = np.ones((4, 4, 3))
        >>> image_to_tensor(img, keepdim=False).shape
        torch.Size([1, 3, 4, 4])
    """
    if len(image.shape) > 4 or len(image.shape) < 2:
        raise ValueError("Input size must be a two, three or four dimensional array")

    input_shape = image.shape
    tensor: torch.Tensor = torch.from_numpy(image)

    if len(input_shape) == 2:
        # (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)
    elif len(input_shape) == 3:
        # (H, W, C) -> (C, H, W)
        tensor = tensor.permute(2, 0, 1)
    elif len(input_shape) == 4:
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)
        keepdim = True  # no need to unsqueeze
    else:
        raise ValueError(f"Cannot process image with shape {input_shape}")

    return tensor.unsqueeze(0) if not keepdim else tensor




def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')    
    

def check_if_there_is_image_files_4_given_extension_when_input_source_is_image_seqeunce(file_or_dir, ext):
    if not os.path.isfile(file_or_dir):
        assert ext is not None, 'Image file extension must be given since the input source is image sequence.'
        li_ext = [strin.strip() for strin in ext.split('_')]
        li_fn = []
        for ext in li_ext:
            li_fn += get_list_of_file_path_under_1st_with_3rd_extension(file_or_dir, False, ext)
        assert li_fn, 'There is NO image files whose extension is {} under {}'.format(li_ext, file_or_dir)   

def is_image_file(fn): 
    ext = (".bmp", ".ppm", ".png", ".gif", ".jpg", ".jpeg", ".tif", ".pgm", ".webp")
    return fn.lower().endswith(ext)


def is_this_existing_directory(path_dir):
    return os.path.exists(path_dir) and os.path.isdir(path_dir)

#########################################################################################################
# input
# direc = '/path/to/the/directory'
# remove_directory(direc)  
def remove_directory(direc):
    return shutil.rmtree(direc)

#########################################################################################################
# input
# direc = '/path/to/the/directory'
# rm_directory_if_exist(direc)  
def rm_directory_if_exist(direc):
    if is_this_existing_directory(direc):
        remove_directory(direc)    

#########################################################################################################
# input
# direc = '/path/to/the/directory'
# rm_and_mkdir(direc)  
def rm_and_mkdir(direc):
    rm_directory_if_exist(direc)
    os.makedirs(direc)



def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  #output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_original: Optional[str] = None,
                  str_rgb_bg: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  is_hair: bool = False,
                  ignore_temporal: bool = False,
                  ext: str = None,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None,
                  is_segmentation: bool = False, 
                  n_iter: int = 10):
    
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    #assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    #assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    #print(f'output_original : {output_original}, output_foreground : {output_foreground}, output_alpha : {output_alpha}, output_composition : {output_composition}');   exit()
    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if not (os.path.isfile(input_source) and is_image_file(input_source)):
        print(f'The given image file is NOT really an image file. Check the file : {input_source}');    exit()
    dir_out = ''.join(input_source.split('.')[:-1]);   rm_and_mkdir(dir_out)
    dir_out_pha = os.path.join(dir_out, 'pha');   rm_and_mkdir(dir_out_pha)
    dir_out_com = os.path.join(dir_out, 'com');   rm_and_mkdir(dir_out_com)
    #print(f'dir_out_pha : {dir_out_pha}');  exit()
    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    
    #if (output_composition is not None) and (output_type == 'video'):
    #if output_composition is not None:
    if str_rgb_bg:
        li_bgr_bg = [int(str_rgb) for str_rgb in str_rgb_bg.split('_')]
        bgr = torch.tensor(li_bgr_bg, device=device, dtype=dtype).div(255).view(1, 3, 1, 1)
    else:     
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 3, 1, 1)
    
    try:
        with torch.no_grad():
            #print('output_composition : {}'.format(output_composition));
            #bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            li_sec_inf = list();    li_sec_total = list()
            li_id = None
            #with Image.open(self.files[idx]) as src:
            with Image.open(input_source) as src:
                src.load()
            if transform is not None:
                src = transform(src)
            src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
#            print(f'src.shape : {src.shape}');  #   torch.Size([1, 3, 1145, 1600]); exit()
            if downsample_ratio is None:
                downsample_ratio = auto_downsample_ratio(*src.shape[2:])
            #print(f'downsample_ratio : {downsample_ratio}');   # 0.32  exit()
            for idx in range(n_iter):
                fgr, pha, *rec = model(src, *rec, downsample_ratio, is_segmentation)
                #print(f'pha.shape : {pha.shape}');  #exit()#   torch.Size([1, 1, 1145, 1600]); exit()
                com = src * pha + bgr * (1 - pha)   #   2nd modification
                #print(f'src.shape : {src.shape}, pha.shape : {pha.shape}, bgr.shape : {bgr.shape}, com.shape : {com.shape}');  exit();#   torch.Size([1, 1, 1145, 1600]); exit()
                #print(f'com.shape : {com.shape}');  exit();#   torch.Size([1, 1, 1145, 1600]); exit()
                to_pil_image(pha[0]).save(os.path.join(dir_out_pha, f'{idx:02d}.png'))
                to_pil_image(com[0]).save(os.path.join(dir_out_com, f'{idx:02d}.png'))
                #to_pil_image(pha[0]).save(os.path.join(self.path, aidi + '.' + self.extension))
                #to_pil_image(com[0]).save(os.path.join(self.path, aidi + '.' + self.extension))
    finally:
        '''
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()
        '''

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


class Converter:
    #def __init__(self, variant: str, checkpoint: str, device: str):
    def __init__(self, variant: str, checkpoint: str, device: str, precision: str):
        print('device :', device);  #exit(0)
        print('torch.cuda.device_count() :', torch.cuda.device_count())
        print('torch.cuda.current_device() :', torch.cuda.current_device())
        print('torch.cuda.get_device_name(torch.cuda.current_device()) :', torch.cuda.get_device_name(torch.cuda.current_device()))
        #t0 = MattingNetwork(variant)
        #t1 = t0.eval()
        #t2 = t1.to(device)
        #self.model = MattingNetwork(variant).eval().to(device)
        self.precision = torch.float32 if precision == 'float32' else torch.float16
        #print(f'self.precision : {self.precision}');    #   float32 exit()
        #print(f'variant : {variant}');    # mobilenetv3 exit()
        self.model = MattingNetwork(variant).eval().to(device, self.precision)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = torch.jit.script(self.model);  self.model = torch.jit.freeze(self.model)
        self.device = device
    
    def convert(self, *args, **kwargs):
        convert_video(self.model, device=self.device, dtype=self.precision, *args, **kwargs)
        #convert_video(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)
    
if __name__ == '__main__':
    import argparse
    from model import MattingNetwork
    '''
    from connected_components import connected_components
    divider = 4 #   which means 150 for saiz of Resize
    n_iter = 30
    img: np.ndarray = cv2.imread("cells_binary.png", cv2.IMREAD_GRAYSCALE)
    #img = img[:int(img.shape[0] / 3)]
    img_t: torch.Tensor = image_to_tensor(img)  # CxHxW
    print('img_t.shape ori : {}'.format(img_t.shape))        #   (1, 602, 602)
    #saiz = [int(img_t.shape[-2] / 2), int(img_t.shape[-1] / 2)]
    size_ori = img_t.shape[-2]
    saiz = int(size_ori / divider)
    print('saiz : {}'.format(saiz))
    img_t = transforms.Resize(size = saiz)(img_t);
    print('img_t.shape 1 : {}'.format(img_t.shape));  #exit(0);
    img_t = img_t[None,...].float() / 255.
    print('img_t.shape 2 : {}'.format(img_t.shape));  #exit(0);
    labels_out = connected_components(img_t, num_iterations=n_iter)
    #labels_out = connected_components(img_t, num_iterations=200)
    print('labels_out.shape b4 : {}'.format(labels_out.shape));  #exit(0);
    print('labels_out.squeeze().shape b4 : {}'.format(labels_out.squeeze().shape));  #exit(0);
    print('labels_out.squeeze() b4 : {}'.format(labels_out.squeeze()));  #exit(0);
    print('torch.unique(labels_out) b4 :'); print(torch.unique(labels_out));   
    labels_out = transforms.Resize(size = size_ori, interpolation = transforms.InterpolationMode.NEAREST)(labels_out);
    print('labels_out.shape after : {}'.format(labels_out.shape));  #exit(0);
    print('labels_out.squeeze().shape after : {}'.format(labels_out.squeeze().shape));  #exit(0);
    print('labels_out.squeeze() after : {}'.format(labels_out.squeeze()));  #exit(0);
    print('torch.unique(labels_out) after :'); print(torch.unique(labels_out));   
    exit(0);
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--input-source', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--downsample-ratio', type=float)
    parser.add_argument('--output-composition', type=str)
    parser.add_argument('--output-original', type=str)
    parser.add_argument('--str-rgb-bg', type=str)
    parser.add_argument('--is_hair', action='store_true')
    parser.add_argument('--ext', type=str)
    parser.add_argument('--output-alpha', type=str)
    parser.add_argument('--output-foreground', type=str)
    #parser.add_argument('--output-type', type=str, required=True, choices=['video', 'png_sequence'])
    parser.add_argument('--n_iter', type=int, default = 20)
    parser.add_argument('--output-video-mbps', type=int, default=1)
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--disable-progress', action='store_true')
    parser.add_argument('--is-segmentation', type=str, default = 'False')
    parser.add_argument('--ignore_temporal', type=str, default = 'False')
    parser.add_argument('--precision', type=str, default = 'float32')
    args = parser.parse_args()
    #converter = Converter(args.variant, args.checkpoint, args.device)
    
    check_if_there_is_image_files_4_given_extension_when_input_source_is_image_seqeunce(
        args.input_source, args.ext)
    
    #print('args.is_hair : {}'.format(args.is_hair));  exit(0)
    #print(f'args.downsample_ratio : {args.downsample_ratio}');  exit(0)
    converter = Converter(args.variant, args.checkpoint, args.device, args.precision)
    converter.convert(
        input_source=args.input_source,
        input_resize=args.input_resize,
        downsample_ratio=args.downsample_ratio,
        #output_type=args.output_type,
        output_composition=args.output_composition,
        output_original = args.output_original,
        str_rgb_bg = args.str_rgb_bg,
        output_alpha=args.output_alpha,
        output_foreground=args.output_foreground,
        output_video_mbps=args.output_video_mbps,
        seq_chunk=args.seq_chunk,
        num_workers=args.num_workers,
        ext = args.ext,
        is_hair = args.is_hair,
        ignore_temporal = str2bool(args.ignore_temporal),
        is_segmentation = str2bool(args.is_segmentation), progress=not args.disable_progress,
        #progress=not args.disable_progress
        n_iter = args.n_iter
    )
    
    
