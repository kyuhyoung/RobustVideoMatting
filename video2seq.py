import os, cv2, sys, shutil

def video_2_image_sequence(fn_vid, dir_seq, xywh_crop):
    cap = cv2.VideoCapture(fn_vid)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file");    exit(0)
    if os.path.exists(dir_seq):
        shutil.rmtree(dir_seq)
    os.makedirs(dir_seq)     
    if xywh_crop:
        x_crop, y_crop, w_crop, h_crop = xywh_crop[0], xywh_crop[1], xywh_crop[2], xywh_crop[3]
    # Read until video is completed
    cnt = 0;    n_img = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     
        if ret == True:
            if xywh_crop:
                frame = frame[y_crop : y_crop + h_crop, x_crop : x_crop + w_crop] 
            fn_img = os.path.join(dir_seq, '%05d.png' % (cnt))
            # save image
            #plt.imsave(OUTPUT_IMAGE_PATH, frame)
            cv2.imwrite(fn_img, frame)
            #print("Now %d-th images being processed..." % (cnt))
            sys.stdout.write('{} / {} th image saved at {}\r'.format(cnt, n_img, fn_img));  sys.stdout.flush();
        # Break the loop
        else: 
            break
        cnt += 1
    # When everything done, release the video capture object
    cap.release()
    return

def main():
    n_arg = len(sys.argv) - 1
    if n_arg < 2:
        print('Usage : python3 video2seq PATH_VIDEO DIR_SEQUENCE [XYWH_CROP]');  exit(0);
    fn_vid = sys.argv[1];   dir_seq = sys.argv[2]
    xywh_crop = []
    if n_arg >= 3:
        xywh_crop = sys.argv[3].split('_')
        if 4 != len(xywh_crop):
            print('The third option given is {}, which is NOT in the form of X_Y_W_H');  exit(0);
        xywh_crop = [int(xywh) for xywh in xywh_crop]
    video_2_image_sequence(fn_vid, dir_seq, xywh_crop)
    return;

if __name__ == '__main__':
    main()

