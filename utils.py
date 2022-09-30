import os
import sys
import cv2
import numpy as np

def get_exact_file_name_from_path(str_path):
    return os.path.splitext(os.path.basename(str_path))[0]

def concatenate_images_from_sequences_into_video(fn_video, li_li_im, li_caption, is_horizontal, fps_output, n_max_per_row_or_col, pxl_max_per_row_or_col = 0, color_bg = (255, 255, 255), wh_interval = (0, 0)):
    
    #   check if the # of images of each directory are the same.
    #li_li_fn_img = [get_list_of_image_path_under_this_directory(direc, ext_img) for direc in li_dir]
    li_n_frm = [len(li_im) for li_im in li_li_im]
    r_all_n_frm_same = are_all_elements_of_list_identical(li_n_frm)
    #print('r_all_n_frm_same :', r_all_n_frm_same);  exit(0);
    if not r_all_n_frm_same:
        print('The # of images in the directories are {}, that is, the # of frames are not the same'.format(li_n_frm));   exit(0) 
    n_frm = li_n_frm[0]
    #   compute the # of rows and # of cols
    li_hwc = [li_im[0].shape for li_im in li_li_im]
    #print('li_hwc :', li_hwc); 
    n_caption = len(li_caption) if li_caption else 0
    li_li_idx = []; li_li_xywh = [] 
    if is_horizontal:
        x_cur = 0;  y_cur = 0;   w_max = 0; h_max = 0;  w_sum = 0;  h_max_cur = -1
        li_idx = [];    li_xywh = []   
        for idx, hwc in enumerate(li_hwc):
            print('idx :', idx) 
            h_cur, w_cur = hwc[0], hwc[1]
            xywh = (x_cur, y_cur, w_cur, h_cur)
            print('xywh :', xywh)
            is_over_pixel = pxl_max_per_row_or_col > 0 and (w_sum + w_cur) > pxl_max_per_row_or_col
            is_over_slot = n_max_per_row_or_col > 0 and (len(li_idx) + 1) > n_max_per_row_or_col
            if is_over_pixel or is_over_slot:
                w_sum -= wh_interval[0]
                w_max = max(w_max, w_sum) 
                h_max += h_max_cur + wh_interval[1]
                li_li_idx.append(li_idx)
                li_li_xywh.append(li_xywh)
                x_cur = 0;  y_cur = h_max  
                xywh = (x_cur, y_cur, w_cur, h_cur)
                li_idx = [idx];    li_xywh = [xywh];  
                w_sum = w_cur + wh_interval[0];  h_max_cur = -1; 
                x_cur = w_sum
            else: 
                w_sum += w_cur + wh_interval[0]
                li_idx.append(idx)
                li_xywh.append(xywh)
                x_cur = w_sum;  
            h_max_cur = max(h_max_cur, h_cur)
        if 0 == h_max and 0 == w_max:
            w_max = w_sum;  
        li_li_idx.append(li_idx);   li_li_xywh.append(li_xywh) 
        h_max += h_max_cur
        w_max -= wh_interval[0];    #h_max -= wh_interval[1]        
    else:
        x_cur = 0;  y_cur = 0;  w_max = 0;  h_max = 0;  h_sum = 0;  w_max_cur = -1
        li_idx = []; li_xywh = []
        for idx, hwc in enumerate(li_hwc):
            print('idx :', idx) 
            h_cur, w_cur = hwc[0], hwc[1]
            xywh = (x_cur, y_cur, w_cur, h_cur)
            print('xywh :', xywh)
            
            is_over_pixel = pxl_max_per_row_or_col > 0 and (h_sum + h_cur) > pxl_max_per_row_or_col
            is_over_slot = n_max_per_row_or_col > 0 and (len(li_idx) + 1) > n_max_per_row_or_col
            if is_over_pixel or is_over_slot:
                h_sum -= wh_interval[1]
                h_max = max(h_max, h_sum)
                w_max += w_max_cur + wh_interval[0]
                li_li_idx.append(li_idx)
                li_li_xywh.append(li_xywh)
                y_cur = 0;  x_cur = w_max
                xywh = (x_cur, y_cur, w_cur, h_cur)
                li_idx = [idx]; li_xywh = [xywh]
                h_sum = h_cur + wh_interval[1]; w_max_cur = -1;
                y_cur = h_sum
            else:
                h_sum += h_cur + wh_interval[1]
                li_idx.append(idx)
                li_xywh.append(xywh)
                y_cur = h_sum
            w_max_cur = max(w_max_cur, w_cur)
        if 0 == w_max and 0 == h_max:
            h_max = h_sum
        li_li_idx.append(li_idx);   li_li_xywh.append(li_xywh)
        w_max += w_max_cur
        h_max -= wh_interval[1]
    print('li_li_idx :', li_li_idx);    print('li_li_xywh :', li_li_xywh);   #exit(0)
    print('h_max :', h_max);    print('w_max :', w_max);    #exit(0)
    #   compute size of entire image
    #   fill the entire image with the given color
    #exit(0) 
     #tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    if int(cv2.__version__[0]) < 3:
        #writer = cv2.VideoWriter(fn_video, cv2.cv.CV_FOURCC(*'MPEG'), fps_output, (w_max, h_max), True)
        writer = cv2.VideoWriter(fn_video, cv2.cv.CV_FOURCC(*'XVID'), fps_output, (w_max, h_max), True)
        #writer = cv2.VideoWriter(fn_video, cv2.cv.CV_FOURCC(*'mp4v'), fps_output, (w_max, h_max), True)
    else:
        #writer = cv2.VideoWriter(fn_video, cv2.VideoWriter_fourcc(*'MPEG'), fps_output, (w_max, h_max), True)
        writer = cv2.VideoWriter(fn_video, cv2.VideoWriter_fourcc(*'XVID'), fps_output, (w_max, h_max), True)
        #writer = cv2.VideoWriter(fn_video, cv2.VideoWriter_fourcc(*'mp4v'), fps_output, (w_max, h_max), True)
    
    if True:
    #if is_horizontal:
        for iF in range(n_frm): 
            sys.stdout.write('iF : {} / {} \r'.format(iF, n_frm));    sys.stdout.flush()
            im = np.zeros((h_max, w_max, 3), np.uint8); 
            im[:, :] = color_bg
            #   for each row
            for li_idx, li_xywh in zip(li_li_idx, li_li_xywh): 
                #   for each col
                for idx, xywh in zip(li_idx, li_xywh):
                #   paste into the region.
                    x_from = xywh[0];   y_from = xywh[1];   x_to = x_from + xywh[2];    y_to = y_from + xywh[3]
                    if idx < n_caption:
                        im[y_from : y_to, x_from : x_to, :] = add_image_text(li_li_im[idx][iF], li_caption[idx])
                    else:    
                        im[y_from : y_to, x_from : x_to, :] = li_li_im[idx][iF]
            #cv2.imwrite('temp.bmp', im); exit(0)
            writer.write(im)
    '''
    else:
        for iF in range(n_frm): 
            sys.stdout.write('iF : {} / {} \r'.format(iF, n_frm));    sys.stdout.flush()
            im = np.zeros((h_max, w_max, 3), np.uint8); 
            im[:, :] = color_bg
            #   for each col
            for li_idx, li_xywh in zip(li_li_idx, li_li_xywh): 
                #   for each row
                for idx, xywh in zip(li_idx, li_xywh):
                #   paste into the region.
                    x_from = xywh[0];   y_from = xywh[1];   x_to = x_from + xywh[2];    y_to = y_from + xywh[3]
                    if idx < n_caption:
                        im[y_from : y_to, x_from : x_to, :] = add_image_text(li_li_im[idx][iF], li_caption[idx])
                    else:    
                        im[y_from : y_to, x_from : x_to, :] = li_li_im[idx][iF]
            #cv2.imwrite('temp.bmp', im); exit(0)
            writer.write(im)
    '''
    writer.release()
    return
 

def are_all_elements_of_list_identical(li):
    return li.count(li[0]) == len(li) 


def str2bool(val):
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        if val.lower() in ['true', 't', 'yes', 'y']:
            return True
        elif val.lower() in ['false', 'f', 'no', 'n']:
            return False
    return False

def is_this_empty_string(strin):
    return (strin in (None, '')) or (not strin.strip())

def get_list_of_file_path_under_1st_with_2nd_extension(direc, ext = ''):
    #print('direc :', direc);    print('ext :', ext);    exit(0)
    li_path_total = []
    is_extension_given = not (is_this_empty_string(ext))
    for dirpath, dirnames, filenames in os.walk(os.path.expanduser(direc)):
        n_file_1 = len(filenames)
        #print('n_file_1 :', n_file_1)
        if n_file_1:
            if is_extension_given:
                li_path = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(ext.lower())]
            else:
                li_path = [os.path.join(dirpath, f) for f in filenames]
            n_file_2 = len(li_path)
            #print('n_file_2 :', n_file_2)
            if n_file_2:
                li_path_total += li_path
            #exit(0)      
    #print('sorted(li_path_total):', sorted(li_path_total)); exit(0)
    return sorted(li_path_total)


def resize_image_sequences(li_li_im, li_wh_resize, color_bg):
    n_seq = len(li_li_im)
    for iS, li_im in enumerate(li_li_im):
        ever_resized = False
        wh_resize = li_wh_resize[iS]   
        shall_resize = 0 < wh_resize[0] or 0 < wh_resize[1]   
        if shall_resize:
            wh_tgt = wh_resize
            n_im_4_this_seq = len(li_im)
            for iI, im in enumerate(li_im):
                wh_src = [li_li_im[iS][iI].shape[1], li_li_im[iS][iI].shape[0]]
                if 0 == wh_resize[0]:
                    wh_tgt[0] = round_i(wh_src[0] * (float(wh_tgt[1]) / float(wh_src[1]))) 
                elif 0 == wh_resize[1]:
                    wh_tgt[1] = round_i(wh_src[1] * (float(wh_tgt[0]) / float(wh_src[0]))) 
                if wh_tgt != wh_src:
                    sys.stdout.write('Resizing {} / {} th image of {} / {} th sequence. From {} to {}\r'.format(iI, n_im_4_this_seq, iS, n_seq, wh_src, wh_tgt));    sys.stdout.flush(); 
                    
                    li_li_im[iS][iI] = letterboxing_opencv(li_li_im[iS][iI], wh_tgt, 'center', -100, means_4_pad = color_bg)
                    ever_resized = True
        if ever_resized:
            print()
    return li_li_im



def concatenate_sequences_from_folders_into_video_with_audio(dir_out, fn_video, li_dir, li_ext, li_caption, li_data_type, li_wh_resize, li_uv_template_fname, li_texture_img_fname, audio_fname, is_horizontal, fps_output, n_max_per_row_or_col, pxl_max_per_row_or_col, color_bg, wh_interval):
    #print('li_dir :', li_dir);  exit(0)    
    if not os.path.exists(dir_out): 
        os.makedirs(dir_out)
    #li_is_mesh = [ext.lower() in ['ply', 'obj'] for ext in li_ext]
    li_is_mesh = ['ply' == ext.lower()[-3:] or 'obj' == ext.lower()[-3:] for ext in li_ext]
    #print('li_is_mesh :', li_is_mesh);  exit(0)
    #li_li_fn = [get_list_of_image_path_under_this_directory(direc, li_ext[iD]) for iD, direc in enumerate(li_dir)]
    print('li_dir :', li_dir);  #exit(0)
    li_li_fn = [get_list_of_file_path_under_1st_with_2nd_extension(direc, li_ext[iD]) for iD, direc in enumerate(li_dir)]
    #print('li_li_fn :', li_li_fn);  #exit(0)
    li_n_frm = [len(li_fn) for li_fn in li_li_fn]
    print('li_n_frm :', li_n_frm);  #exit(0)
    r_all_n_frm_same = are_all_elements_of_list_identical(li_n_frm)
    if not r_all_n_frm_same:
        print('The # of images in the directories {} are {}, that is, the # of frames are not the same'.format(li_dir, li_n_frm));   exit(0) 
    n_seq = len(li_n_frm)
    li_li_im = [None] * n_seq
    center = None
    for iS, is_mesh in enumerate(li_is_mesh):
        if is_mesh:
            dir_mesh = li_dir[iS]
            if not li_data_type[iS]:
                print('Data_type should be given, once it is mesh sequence. Data_type for {} was not given !!'.format( dir_mesh));   exit(0) 
            postfix_dir = dir_mesh[-7:]
            print('postfix_dir :', postfix_dir);    #exit(0)
            if '_meshes' == postfix_dir:
                fn_only = get_exact_file_name_from_path(dir_mesh[:-7])
                #dir_img_out = dir_mesh[:-7] + '_imgs'
                #print('dir_img_out :', dir_img_out)    #exit(0)
            else:
                fn_only = get_exact_file_name_from_path(dir_mesh)
            dir_img_out = os.path.join(dir_out, fn_only + '_imgs')
            if not os.path.exists(dir_img_out):
                os.makedirs(dir_img_out)
            if li_uv_template_fname and li_texture_img_fname:
                if os.path.exists(li_uv_template_fname[iS]) and os.path.exists(li_texture_img_fname[iS]):
                    uv_template = Mesh(filename = li_uv_template_fname[iS])
                    vt, ft = uv_template.vt, uv_template.ft
                    tex_img = cv2.imread(li_texture_img_fname[iS])[:, :, ::-1]
                else:
                    vt, ft, tex_img = None, None, None
            else:
                vt, ft, tex_img = None, None, None
            #print('len(li_li_fn[iS]) :', len(li_li_fn[iS])); #exit(0) 
            li_li_im[iS], center, _ = render_mesh_file_sequence(li_li_fn[iS], dir_img_out, li_data_type[iS], center, bg_mesh_rgb, None, vt, ft, tex_img)
            #print('len(li_li_im[iS]) :', len(li_li_im[iS])); exit(0) 
        else:
            li_li_im[iS] = [cv2.imread(fn_img) for fn_img in li_li_fn[iS]] 
    #li_li_im = [[cv2.imread(fn_img) for fn_img in li_fn_img] for li_fn_img in li_li_fn_img] 
    if li_wh_resize:
        li_li_im = resize_image_sequences(li_li_im, li_wh_resize, color_bg)
    #path_video = os.path.join(dir_out, fn_video + '.mp4')
    path_video = os.path.join(dir_out, fn_video + '.mkv')
    concatenate_images_from_sequences_into_video(path_video, li_li_im, li_caption, is_horizontal, fps_output, n_max_per_row_or_col, pxl_max_per_row_or_col, color_bg, wh_interval)
    if audio_fname:
        video_fname = os.path.join(dir_out, '{}_w_audio.mp4'.format(fn_video)) #if postphix else os.path.join(out_path, 'w_audio.mp4')
        cmd = ('ffmpeg' + ' -y -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
            audio_fname, path_video, video_fname)).split()
        call(cmd)
        print('Saved rendered sequence at {}'.format(video_fname))



