import sys, os, shutil

def is_this_empty_string(strin):
    return (strin in (None, '')) or (not strin.strip())


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


def sort_1st_according_to_2nd(li_1st, li_2nd):
    return [e_1 for _, e_1 in sorted(zip(li_2nd, li_1st))]

def get_exact_file_name_from_path(str_path):
    return os.path.splitext(os.path.basename(str_path))[0]

def is_only_number(strin):
    return strin.isdigit()

def check_if_files_under_1st_with_2nd_extension_are_sequential_wo_hole(direc, ext):
    are_sequential_wo_hole = True
#   get list of file names with given extension
    li_path = get_list_of_file_path_under_1st_with_3rd_extension(direc, False, ext)
    #print('li_path :', li_path);    exit(0)
#   sort the file names
    li_idx = []
    for path_file in li_path:
        fn_only = get_exact_file_name_from_path(path_file)
        if not is_only_number(fn_only):
            print('One of the file name uner {} is {}, which is NOT number !!'.format(direc, path_file));  
            are_sequential_wo_hole = False
            return are_sequential_wo_hole, None, None, None
        li_idx.append(int(fn_only))
    n_file_physically = len(li_idx)
    #print('n_file_physically :', n_file_physically);    exit(0)
    li_path_sorted = sort_1st_according_to_2nd(li_path, li_idx)
    li_idx_sorted = sorted(li_idx)
#   check if there is no hole in the file name list 
    for index in range(n_file_physically - 1):
        dif = li_idx_sorted[index + 1] - li_idx_sorted[index]    
        if 1 != dif:
            print('Two sequential file names under {} is {} and {}, which is NOT really sequential !!'.format(direc, li_path_sorted[index], li_path_sorted[index + 1]));   #exit(0)
            are_sequential_wo_hole = False
            return are_sequential_wo_hole, None, None, None
#   count the number of files
    n_file_from_file_name = li_idx_sorted[-1] - li_idx_sorted[0] + 1
#   check if the numbers of files are the same
    if n_file_physically != n_file_from_file_name:
        print('The # of physical files is {} and the # of files from file names is {}. Those are NOT the same !!'.format(n_file_physically, n_file_from_file_name));  
        are_sequential_wo_hole = False
        return are_sequential_wo_hole, None, None, None
    di_idx_path = {idx:li_path_sorted[index] for index, idx in enumerate(li_idx_sorted)}
    return are_sequential_wo_hole, li_path_sorted, li_idx_sorted, di_idx_path      


def resample_integers_btn_1st_2nd_to_get_3rd_number(i_from, i_to, num):
    
    #i_from = 10;    i_to = 19;  num = 20
    li_i = [None] * num
    n_i = i_to - i_from + 1
    if n_i == num:
        li_i = range(i_from, i_to + 1)
    else:
        ratio = float(n_i) / float(num)
        for ii in range(num):
            delta = ratio * float(ii)
            li_i[ii] = int(float(i_from) + delta) 
    #print('li_i :', li_i);  exit(0)
    return li_i
        

def resample_files_under_1st_with_2nd_extension_to_get_3rd_number_of_files(dir_in, ext, n_file_to_be, dir_out, idx_from = 0, idx_to = -1, idx_from_new = None):
    are_sequential_wo_hole, li_path_sorted, li_num_sorted, di_num_path = check_if_files_under_1st_with_2nd_extension_are_sequential_wo_hole(dir_in, ext)
    '''
    print('are_sequential_wo_hole :', are_sequential_wo_hole);
    print('li_path_sorted :', li_path_sorted);  #exit(0)
    print('li_num_sorted :', li_num_sorted);    #exit(0)
    print('di_num_path :', di_num_path);        exit(0)
    print('idx_from :', idx_from);        #exit(0)
    print('idx_to :', idx_to);        exit(0)
    '''
    li_num_new = resample_integers_btn_1st_2nd_to_get_3rd_number(li_num_sorted[idx_from], li_num_sorted[idx_to], n_file_to_be) 
    #print('idx_from_new :', idx_from_new);  exit(0)
    if idx_from_new is not None:
        idx_base = idx_from_new
    else:
        idx_base = li_num_sorted[idx_from]
#   for each new file
    #print('idx_base :', idx_base);  exit(0)
    for index, num_new in enumerate(li_num_new):
#       make file name
        idx_new = idx_base + index
        fn_out = os.path.join(dir_out, '{:05d}.{}'.format(idx_new, ext))
#       copy file
        shutil.copyfile(di_num_path[num_new], fn_out)
        print('Copied a file {} into {}'.format(di_num_path[num_new], fn_out))

def mkdir_after_rm_if_exist(direc):
    if os.path.exists(direc):
        shutil.rmtree(direc)
    os.makedirs(direc)

def main():
    dir_in = sys.argv[1]
    if not os.path.exists(dir_in):
        print('The given file directory {} does NOT exist. Please check !!'.format(dir_in));   exit(0)
    ext = sys.argv[2]
    n_file_to_be = int(sys.argv[3])
    dir_out = sys.argv[4]
    mkdir_after_rm_if_exist(dir_out)
    n_arg = len(sys.argv)
    #print('n_arg :', n_arg);    exit(0)
    idx_from = 0;   idx_to = -1
    if n_arg >= 7:
        idx_from = int(sys.argv[5])
        idx_to = int(sys.argv[6])
    idx_from_new = idx_from      
    if n_arg >= 8:
        idx_from_new = int(sys.argv[7])

    resample_files_under_1st_with_2nd_extension_to_get_3rd_number_of_files(dir_in, ext, n_file_to_be, dir_out, idx_from, idx_to, idx_from_new)  


if __name__ == '__main__':
    main()


