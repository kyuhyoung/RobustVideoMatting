import os, sys, cv2, shutil
import numpy as np

def get_mask_of_given_color(im_bgr, bgr):
    min_bgr = np.array(bgr) - 3
    max_bgr = np.array(bgr) + 3
    #print('min_bgr :', min_bgr);    print('max_bgr :', max_bgr);    exit(0)
    return cv2.inRange(im_bgr, min_bgr, max_bgr)

def is_this_empty_string(strin):
    return (strin in (None, '')) or (not strin.strip())

def mkdir_after_rm_if_exist(direc):
    if os.path.exists(direc):
        shutil.rmtree(direc)
    os.makedirs(direc)




def get_list_of_file_path_under_1st_with_3rd_extension(direc, include_subdirectories, ext = ''):
    li_path_total = []
    is_extension_given = not (is_this_empty_string(ext))
    if include_subdirectories:
        for dirpath, dirnames, filenames in os.walk(os.path.expanduser(direc)):
            n_file_1 = len(filenames)
            if n_file_1:
                if is_extension_given:
                    li_path = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith(ext.lower())]
                else:
                    li_path = [os.path.join(dirpath, f) for f in filenames]
                n_file_2 = len(li_path)
                if n_file_2:
                    li_path_total += li_path
    else:
        for name_file_dir in os.listdir(direc):
            #print('name_file_dir :', name_file_dir)
            path_file_dir = os.path.join(direc, name_file_dir)
            if os.path.isfile(path_file_dir):
                if is_extension_given:
                    if name_file_dir.lower().endswith(ext.lower()):
                        li_path_total.append(path_file_dir)
                else:
                    li_path_total.append(path_file_dir)
    return sorted(li_path_total)

def get_exact_file_name_from_path(str_path):
    return os.path.splitext(os.path.basename(str_path))[0]




dir_in = sys.argv[1]
if not os.path.exists(dir_in):
    print('The given file directory {} does NOT exist. Please check !!'.format(dir_in));   exit(0)
ext = sys.argv[2]
dir_out = sys.argv[3]
bgr_mask = (int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
kernel_size = int(sys.argv[7])
n_arg = len(sys.argv)
#print('n_arg :', n_arg);    exit(0);
#print('kernel_size :', kernel_size);    exit(0);
       

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size , kernel_size))

#   get list of green file
li_fn_seg = get_list_of_file_path_under_1st_with_3rd_extension(dir_in, False, ext)
li_fn_seg = sorted(li_fn_seg)
n_seg = len(li_fn_seg);
#print('li_fn :', li_fn);    exit(0)
#print('dir_out :', dir_out);    exit(0)
shall_mask_ori = False; dir_ori = None; li_fn_ori = None; im_black = None
if n_arg >= 9:
    dir_ori = sys.argv[8]
    if os.path.exists(dir_ori):
        shall_mask_ori = True
        li_fn_ori = get_list_of_file_path_under_1st_with_3rd_extension(dir_ori, False, ext)
        li_fn_ori = sorted(li_fn_ori)
        n_ori = len(li_fn_ori)
        if n_ori != n_seg:
            print('The # of segmentation file is {} and the # of original file is {}, which is NOT the same !!'.format(n_seg, n_ori)); exit(0)
        im_ori = cv2.imread(li_fn_ori[0]);  
        im_black = np.zeros(im_ori.shape, np.uint8);
        #print('im_black.shape :', im_black.shape);  exit(0)

mkdir_after_rm_if_exist(dir_out)
#   for each green file
for idx, fn_seg in enumerate(li_fn_seg):
#       read file
    im_bgr_ori = cv2.imread(fn_seg)
#       find non-green pixel mask
    mask_ori = get_mask_of_given_color(im_bgr_ori, bgr_mask)
    mask_ori = 255 - mask_ori
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask_ori)
    #print('min_val : {}, max_val : {}, min_loc : {}, max_loc : {}'.format(min_val, max_val, min_loc, max_loc))
    #print('mask_ori.shape :', mask_ori.shape);  exit(0)
#       dilate
    mask_dilate = cv2.dilate(mask_ori, kernel, iterations = 1)
    fn_only = get_exact_file_name_from_path(fn_seg)
    if shall_mask_ori:
        #im_ori = cv2.imread(li_fn_ori[idx[)
        im_bgr_dilated_mask = cv2.imread(li_fn_ori[idx])
        im_bgr_dilated_mask[np.where(255 != mask_dilate)] = 0 
        fn_bgr_dialated_mask = os.path.join(dir_out, fn_only + '.' + ext)
        cv2.imwrite(fn_bgr_dialated_mask, im_bgr_dilated_mask)
        print('[{} / {}] saved original with dilated mask at {}'.format(idx, n_ori, fn_bgr_dialated_mask))
    else:    
        fn_mask = os.path.join(dir_out, fn_only + '.' + ext)
        cv2.imwrite(fn_mask, mask_dilate)
        print('[{} / {}] saved dilated mask at {}'.format(idx, n_seg, fn_mask))
        
    #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask_dilate)
    #print('min_val : {}, max_val : {}, min_loc : {}, max_loc : {}'.format(min_val, max_val, min_loc, max_loc))
    #print('mask_ori.shape :', mask_ori.shape);  exit(0)
    #cv2.imwrite('im_bgr_ori.png', im_bgr_ori);  
    #cv2.imwrite('mask_ori.png', mask_ori);  cv2.imwrite('mask_dilate.png', mask_dilate);    exit(0)
#       make fn
    #print('fn_mask : {}'.format(fn_mask));  exit(0)
#       save
