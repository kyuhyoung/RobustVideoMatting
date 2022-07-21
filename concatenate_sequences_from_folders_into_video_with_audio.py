import argparse
#from utils.util import concatenate_sequences_from_folders_into_video_with_audio, str2bool
from utils import concatenate_sequences_from_folders_into_video_with_audio, str2bool

parser = argparse.ArgumentParser(description='Concatenate image sequences into video as well as audio')
parser.add_argument('--fn_vid', help='Path to the resulted video file', required = True)
parser.add_argument('--dir_out', help='Directory to save result video', required = True)
parser.add_argument('--li_ext', nargs = '+', help='list of image extensions', required = True)
parser.add_argument('--li_dir', nargs = '+', help='list of image directories', required = True)
parser.add_argument('--li_caption', nargs = '+', help='list of captions')
parser.add_argument('--li_data_type', nargs = '+', help='list of data_type')
parser.add_argument('--li_wh_resize', nargs = '+', help='list of sizes for resizing. For example, 800_600 0_400 means the images of the first sequence are letterboxed as 800 by 600 and the images of the second sequence are resized so that the height is 400 and the width is scaled accordingly')
parser.add_argument('--li_uv_template_fname', default = None, nargs = '+', help='list of uv_template_fname')
parser.add_argument('--li_texture_img_fname', default = None, nargs = '+', help='list of texture_img_fname')
parser.add_argument('--fn_aud', help='Path to audio file')
#parser.add_argument('--is_horizontal', default = 'True', help='concatenate horizontally if ON')
parser.add_argument('--is_horizontal', help='concatenate horizontally if ON')
parser.add_argument('--max_slot', type = int, default = 3, help='Max # of slot')
parser.add_argument('--max_pxl', type = int, default = 2500, help='Max # of pixel')
parser.add_argument('--fps_output', type = float, default = 30.0, help='fps of result video')
parser.add_argument('--bg_mesh_rgb', nargs = '+', default='255 255 255', help = 'Background color RGB of mesh, that is, 255 255 255 for white and 0 0 0 for black') 

args = parser.parse_args()

fn_video = args.fn_vid
li_dir_img = args.li_dir
li_ext = args.li_ext
fn_audio = args.fn_aud
dir_out = args.dir_out
li_caption = args.li_caption
is_horizontal = str2bool(args.is_horizontal)
max_slot = args.max_slot
max_pxl = args.max_pxl
fps_output = args.fps_output
li_uv_template_fname = args.li_uv_template_fname
li_texture_img_fname = args.li_texture_img_fname
li_data_type = args.li_data_type
print('args.li_wh_resize :', args.li_wh_resize)
n_seq = len(li_dir_img)
if args.li_wh_resize:
    li_li_wh_resize = [wh_resize.split('_') for wh_resize in args.li_wh_resize]
    print('li_li_wh_resize :', li_li_wh_resize);  #exit(0)
    if len(li_li_wh_resize) != n_seq:
        print('The # of sequences is {}. However, the # of wh_resize is {}. Please take care of this !!'.format(n_seq, len(li_li_wh_resize)));   exit(0) 
    else:
        li_wh_resize = []
        for li_str_wh_resize in li_li_wh_resize:
            if 2 != len(li_str_wh_resize):
                print('The given li_wh_resize is {}. Please correct this !!'.format(arg.li_wh_resize)); exit(0) 
            li_wh_resize.append([int(str_wh) for str_wh in li_str_wh_resize])    
else:
    li_wh_resize = None
print('li_wh_resize :', li_wh_resize);  #exit(0)
#bg_mesh_rgb = [int(val) for val in args.bg_mesh_rgb]
#print('li_caption :', li_caption);  exit(0)

concatenate_sequences_from_folders_into_video_with_audio(dir_out, fn_video, li_dir_img, li_ext, li_caption, li_data_type, li_wh_resize, li_uv_template_fname, li_texture_img_fname, fn_audio, is_horizontal, fps_output, max_slot, max_pxl, (0, 0, 0), (0, 0))

