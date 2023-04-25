: << 'END'
#################################################################################################
#
#   train 
#
#   original training. 
#
python3 train.py --model-variant mobilenetv3 --dataset videomatte --resolution-lr 512 --seq-length-lr 15 --learning-rate-backbone 0.0001 --learning-rate-aspp 0.0002 --learning-rate-decoder 0.0002 --learning-rate-refiner 0 --checkpoint-dir checkpoint/stage1 --log-dir log/stage1 --epoch-start 0 --epoch-end 20 --is_hair 
#python3 train.py --model-variant mobilenetv3 --dataset videomatte --resolution-lr 512 --seq-length-lr 15 --learning-rate-backbone 0.0001 --learning-rate-aspp 0.0002 --learning-rate-decoder 0.0002 --learning-rate-refiner 0 --checkpoint-dir checkpoint/stage1 --log-dir log/stage1 --epoch-start 0 --epoch-end 20 --checkpoint-save-interval 10 --disable-validation  
END

: << 'END'
#################################################################################################
#   inference 
#
#   inference on video and save to video. 
#
#python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --input-source butter.mp4 --output-type video --output-composition composition_mbps_05.mp4 --output-video-mbps 5   
END

: << 'END'
#   inference on video and save the composition to image seq.
#
#vid_title=butter
#python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --input-source ${vid_title}.mp4 --output-type png_sequence --output-composition ${vid_title}_composition_imgs --str-rgb-bg 255_0_0
END


: << 'END'
#   inference on video and save the original and composition to image seq.
#vid_title=butter
#vid_title=long_hair_youtube
#seq_id=tomotomo_1920_1080
#seq_id=speech_1280_720
#seq_id=fxgear_220629_1080_1920
#seq_id=kinect_1920_1080_1
#seq_id=kinect_1920_1080_2
#dir_data=/data/test_seq/test_vids
declare -A di_scale_width
declare -A di_scale_height
di_scale_width[ori]="1920"
di_scale_height[ori]="1080"
di_scale_width[half]="960"
di_scale_height[half]="540"
di_scale_width[quater]="480"
di_scale_height[quater]="270"
#for seq_id in kinect_1920_1080_1 kinect_1920_1080_2
#for seq_id in kinect_1920_1080_1
#for seq_id in kinect_1920_1080_3 kinect_1920_1080_4
#for idx_seq in 3 4
ext=png
for idx_seq in 2
do
    seq_id=kinect_1920_1080_${idx_seq}
    #dir_data=output/kinect/${seq_id}/modified
    dir_data=/data/test_seq/kinect/${seq_id}/modified/kinect_rgb_${idx_seq}/imgs_ori
    #for input_resize in ori half quater 
    #for input_resize in half 
    #for input_resize in quater 
    for input_resize in ori 
    do
        w_input=${di_scale_width[${input_resize}]}
        h_input=${di_scale_height[${input_resize}]}
        #for down_ratio in 1.0 0.5 0.25
        #for down_ratio in 1.0
        for down_ratio in 0.25
        do
            #for precision in float32 float16
            for precision in float32
            #for precision in float16
            do
                #for is_segmentation in False True
                #for is_segmentation in True
                for is_segmentation in False
                do
                    dir_out=output/${seq_id}_input_resize_${w_input}_${h_input}_downsample_ratio_${down_ratio}_precision_${precision}_is_segmentation_${is_segmentation}
                    python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda --input-source ${dir_data} --ext ${ext} --output-type png_sequence --output-composition ${dir_out}/imgs_comp --str-rgb-bg 255_0_0 --input-resize ${w_input} ${h_input} --downsample-ratio ${down_ratio} --precision ${precision} --is-segmentation ${is_segmentation}
                done
            done    
        done
    done
done    
#python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --input-source ${dir_data}/rgb_dilated_mask_${seq_id}/imgs_ori --output-type png_sequence --output-composition ${dir_out}_rgb_dilated_mask/imgs_comp --str-rgb-bg 0_255_0
#w_input=960
#h_input=540
#dir_out=output/${seq_id}_input_resized_${w_input}_${h_input}
#python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --input-source ${dir_data}/rgb_${seq_id}/imgs_ori --output-type png_sequence --output-composition ${dir_out}/imgs_comp --str-rgb-bg 0_255_0 --input-resize ${w_input} ${h_input}
#python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --input-source ${dir_data}/${seq_id}/imgs_ori --output-type video --output-composition ${dir_out}/imgs_comp --output-original ${dir_out}/imgs_ori --str-rgb-bg 255_0_0
#python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --input-source ${dir_data}/vid_ori/${seq_id}.mp4 --output-type png_sequence --output-composition ${dir_out}/imgs_comp --output-original ${dir_out}/imgs_ori --str-rgb-bg 255_0_0
END
#
: << 'END'
#   inference RVM hair version on image seq. and save the composition to image seq.
#vid_title=long_hair_youtube_21
#vid_title=fxgear_220629_1080_1920
#vid_title=sooji_1280_720
vid_title=joonhan_1280_720
#vid_title=tomotomo_800_999_1920_1080
#checkpoint=rvm_mobilenetv3.pth
#checkpoint=checkpoint/stage4/ep_12_ib_01500_val_loss_0.12242806323190356.pth
ext=png

checkpoint=checkpoint/stage2/ep_08_ib_00336_val_loss_0.08507319947178603.pth
#for vid_title in joonhan_1280_720 eunchae_1280_720
for vid_title in joonhan_1280_720
do
    dir_in=/data/test_seq/${vid_title}/imgs_ori
    #for down_ratio in 0.25 1.0
    #for down_ratio in 1.0
    for down_ratio in 0.25
    #for down_ratio in 0.125
    do
        #for seg in False True
        #for seg in True
        for seg in False
        do    
            #for ignore_temporal in True False    
            #for ignore_temporal in False    
            for ignore_temporal in True    
            do
                dir_out=output/${vid_title}_downsample_ratio_${down_ratio}_is_segmentation_${seg}_ignore_temporal_${ignore_temporal}/rvm_comp
                python3 inference.py --variant mobilenetv3 --checkpoint ${checkpoint} --device cuda:0 --downsample-ratio ${down_ratio} --ext ${ext} --input-source ${dir_in} --output-type png_sequence --output-composition ${dir_out} --str-rgb-bg 255_0_0 --is-segmentation ${seg} --ignore_temporal ${ignore_temporal} --is_hair 
            done
        done
    done
done    
END

: << 'END'
#   inference on image seq. and save the composition and alpha to image seq.
#dir_test=test/${vid_title}
#dir_1=/data/k-hairstyle/Training/0002.mqset_mini
#dir_1=/data/k-hairstyle/Training/0002.mqset
#dir_1=/data/k-hairstyle/Training/0001.hqset
#dir_1=/data/k-hairstyle/Validation/0002.mqset
#dir_1=/data/k-hairstyle/Validation/0001.hqset
#command_1="find ${dir_1} -mindepth 1 -maxdepth 1 -type d"
ext=jpg_jpeg
#ext=jpeg
#ext=jpeg
#echo "command_1 : ${command_1}"
#res_1=`${command_1}`
#echo "result of command_1 : " ${res_1}
#exit
#for d_resol in 0001.hqset 0002.mqset
#for d_resol in 0001.hqset
#for d_resol in 0001.hqset_5
for d_resol in 0003.rawset_modified_shit
do
    #for d_tra_val in Training Validation
    for d_tra_val in Training
    #for d_tra_val in Validation
    do
        #dir_1=/data/k-hairstyle/Training/0001.hqset
        dir_1=/data/k-hairstyle/${d_tra_val}/${d_resol}
        command_1="find ${dir_1} -mindepth 1 -maxdepth 1 -type d"

        for d_hair_style in `${command_1}`
        #for d_hair_style in 0022
        do  
            echo "d_hair_style : ${d_hair_style}"
            command_2="find ${d_hair_style} -mindepth 1 -maxdepth 1 -type d"
            for d_id in `${command_2}`
            #for d_id_only in 2503.JS218573 2738.JS376405 0775.CP500344 3232.JS634122 2283.JS048924
            do
                #d_id=${dir_1}/${d_hair_style}/${d_id_only}
                echo "d_id : ${d_id}"
                #python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --ext ${ext} --input-source ${d_id} --output-type png_sequence --output-composition ${d_id}/rvm_comp --output-alpha ${d_id}/rvm_alpha --str-rgb-bg 255_0_0
                python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --ext ${ext} --input-source ${d_id} --output-type png_sequence --output-alpha ${d_id}/rvm_alpha --str-rgb-bg 255_0_0
            done
        done  
    done
done    
END


: << 'END'
#   inference on image seq. and save the composition and alpha to image seq.
#dir_test=test/${vid_title}
#dir_1=/data/k-hairstyle/Training/0002.mqset_mini
#dir_1=/data/k-hairstyle/Training/0002.mqset
#dir_1=/data/k-hairstyle/Training/0001.hqset
#dir_1=/data/k-hairstyle/Validation/0002.mqset
#dir_1=/data/k-hairstyle/Validation/0001.hqset
#command_1="find ${dir_1} -mindepth 1 -maxdepth 1 -type d"
#ext=jpg_jpeg
#ext=jpeg
ext=png
#echo "command_1 : ${command_1}"
#res_1=`${command_1}`
#echo "result of command_1 : " ${res_1}
#exit
#for d_resol in 0001.hqset 0002.mqset
#for d_resol in 0001.hqset
#for d_resol in 0001.hqset_5
for aidi in taejeong1_1080_1920 taejeong2_1080_1920 taejeong3_1080_1920 sohyun1_1080_1920 sohyun2_1080_1920  
#for aidi in taejeong1_1080_1920  
do
    d_id=/data/matting/test_seq/fxgear_5th_floor_221207/img_seq/${aidi} 
    for scale in 0.25
    do
        python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --ext ${ext} --input-source ${d_id}/imgs_ori --output-type png_sequence --output-composition ${d_id}/rvm_comp_scale_${scale} --output-alpha ${d_id}/rvm_alpha_scale_${scale} --str-rgb-bg 255_0_0 --downsample-ratio ${scale}
                #python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --ext ${ext} --input-source ${d_id} --output-type png_sequence --output-alpha ${d_id}/rvm_alpha --str-rgb-bg 255_0_0
    done
done    
END

#: << 'END'
#   inference on image seq. and save the composition and alpha to image seq.
#dir_test=test/${vid_title}
#dir_1=/data/k-hairstyle/Training/0002.mqset_mini
#dir_1=/data/k-hairstyle/Training/0002.mqset
#dir_1=/data/k-hairstyle/Training/0001.hqset
#dir_1=/data/k-hairstyle/Validation/0002.mqset
#dir_1=/data/k-hairstyle/Validation/0001.hqset
#command_1="find ${dir_1} -mindepth 1 -maxdepth 1 -type d"
#ext=jpg_jpeg
#ext=jpeg
ext=png
#echo "command_1 : ${command_1}"
#res_1=`${command_1}`
#echo "result of command_1 : " ${res_1}
#exit
#for d_resol in 0001.hqset 0002.mqset
#for d_resol in 0001.hqset
#for d_resol in 0001.hqset_5
dir_root=/data/matting/tmp/hand_video_seqs_png
dir_out=./output/hand_video_png
#cmd_1="find ${dir_root} -type f -name '*.${ext}' -printf '%h\n' | sort -u | sed 's|^|$PWD/|'"
cmd_1="ls -d ${dir_root}/*/"
for d_id in `${cmd_1}`
do
	aidi=$(basename ${d_id})
	#echo "aidi : ${aidi}";	exit;
	#d_id=${dir_root}/${aidi}
	#echo "d_id : ${d_id}";	exit;
	d_out=${dir_out}/${aidi}
	#echo "d_out : ${d_out}";	exit;
	mkdir -p ${d_out}
	cmd_2="ls -d ${d_id}/*/"
	for d_img in `${cmd_2}`
	do
		#echo "d_img : ${d_img}";	exit;
		#d_img=${d_id}/${idx}
		idx=$(basename ${d_img})
		#echo "idx : ${idx}";	exit;
		d_out_comp=${d_out}/${idx}/com
		mkdir -p ${d_out_comp}
		python3 inference.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --ext ${ext} --input-source ${d_img} --output-type png_sequence --output-composition ${d_out_comp} --str-rgb-bg 255_0_0
		exit
	done
done    
#END

 

: << 'END'
#   inference on image seq. and save the composition and alpha to image seq.
#for dir in people_waving_hands_over_1024 people_victory_gesture_and_thumb_up_over_1024 people_ok_hi5_hanads_up_over_1024
for dir in people_victory_gesture_and_thumb_up_over_1024 people_ok_hi5_hands_up_over_1024
#for dir in people_victory_gesture_and_thumb_up_over_1024
do
	dir_img=/data/matting/${dir}
	for phile in ${dir_img}/*
	do
    	echo "phile : ${phile}"
    	if [[ "${phile}" =~ \.(jpg|jpeg|png|webp|gif)$ ]]; then
    	#if [[ "${phile}" =~ \.(webp)$ ]]; then
        	#exit
        	python3 inference_single_image_vs_pseudo_sequence.py --variant mobilenetv3 --checkpoint rvm_mobilenetv3.pth --device cuda:0 --input-source "${phile}" --str-rgb-bg 255_0_0 --n_iter 50
        	#exit    
    	fi
	done
done  
END
    


    

: << 'END'
#################################################################################################
#
#   concatenate multiple image sequences into a video 
#
#   original from voca_korean
#PYOPENGL_PLATFORM=osmesa python3 concatenate_sequences_from_folders_into_video_with_audio.py --max_pxl 3000 --max_slot 2 --dir_out ${out_path} --fn_vid sentence${sentence}_cam_mesh --bg_mesh_rgb 0 0 0 --li_wh_resize 800_800 0_0 --li_data_type VOCA VOCA --li_ext 26_C.jpg ply --li_dir ${dir_data}/multi_cam_imgs/${speaker}/sentence${sentence} ${dir_data}/registereddata/${speaker}/sentence${sentence} --li_caption cam render_registered --fn_aud ${dir_data}/audio/${speaker}/sentence${sentence}.wav
#
#   horizontal
#vid_title=long_hair_youtube_21
#dir_test=test/${vid_title}
#python3 concatenate_sequences_from_folders_into_video_with_audio.py --fps_output 10.0 --is_horizontal True --max_pxl 3000 --max_slot 2 --dir_out ${dir_test}/vid_comp --fn_vid ${vid_title}_ori_comp --li_ext png png --li_dir ${dir_test}/imgs_ori ${dir_test}/imgs_comp
#   vertical
python3 concatenate_sequences_from_folders_into_video_with_audio.py --fps_output 10.0 --is_horizontal False --max_pxl 3000 --max_slot 2 --dir_out ${dir_out}/vid_comp --fn_vid ${seq_id}_ori_comp --li_ext png png --li_dir ${dir_data}/imgs_ori ${dir_out}/imgs_comp
END

: << 'END'
#################################################################################################
#
#   save video into image sequences 
#
#python3 video2seq.py input/fxgear_220629_1080_1920_snapchat.mp4 output/fxgear_220629_1080_1920_snapchat 584_0_608_1080 
#python3 video2seq.py input/sooji_1280_720_snapchat.mp4 output/sooji_1280_720_snapchat
#python3 video2seq.py input/eunchae_1280_720_snapchat.mp4 output/eunchae_1280_720_snapchat
#python3 video2seq.py input/joonhan_1280_720_snapchat.mp4 output/joonhan_1280_720_snapchat
#python3 video2seq.py input/tomotomo_snapchat.mp4 output/tomotomo_snapchat
#python3 video2seq.py input/longhair_snapchat.mp4 output/longhair_snapchat 358_0_1080_1080
#python3 video2seq.py input/sooji_1280_720_youcamvideo.mp4 output/sooji_1280_720_youcamvideo
#python3 video2seq.py input/eunchae_1280_720_youcamvideo.mp4 output/eunchae_1280_720_youcamvideo
#python3 video2seq.py input/joonhan_1280_720_youcamvideo.mp4 output/joonhan_1280_720_youcamvideo
#python3 video2seq.py input/tomotomo_youcamvideo.mp4 output/tomotomo_youcamvideo
#python3 video2seq.py input/kinect_rgb_1.mp4 output/kinect_rgb_1/imgs_ori
for idx_seq in 3 4
#for idx_seq in 3
do
    id_seq=kinect_1920_1080_${idx_seq}
    for id_rgb_seg in kinect_rgb_${idx_seq} kinect_seg_${idx_seq}   
    #for id_rgb_seg in kinect_rgb_${idx_seq}   
    do
        python3 video2seq.py input/${id_rgb_seg}.mp4 /data/${id_seq}/ori/${id_rgb_seg}/imgs_ori
    done
done    

END

: << 'END'
#################################################################################################
#
#   resample files  
#
#python3 resample_files_under_directory.py output/kinect/seq_1/ori/kinect_rgb_1/imgs_ori png 424 output/kinect/seq_1/modified_1/kinect_rgb_1/imgs_ori 10 433 0
#python3 resample_files_under_directory.py output/kinect/seq_1/ori/kinect_seg_1/imgs_ori png 424 output/kinect/seq_1/modified_1/kinect_seg_1/imgs_ori 6 414 0
#python3 resample_files_under_directory.py output/kinect/seq_2/ori/kinect_rgb_2/imgs_ori png 368 output/kinect/seq_2/modified/kinect_rgb_2/imgs_ori 4 371 0
python3 resample_files_under_directory.py output/kinect/seq_2/ori/kinect_seg_2/imgs_ori png 368 output/kinect/seq_2/modified/kinect_seg_2/imgs_ori 1 350 0
END

: << 'END'
#################################################################################################
#
#   make binary mask out of green screen image  
#
#seq_id=kinect_1920_1080_1
seq_id=kinect_1920_1080_2
#python3 resample_files_under_directory.py output/kinect/seq_1/ori/kinect_rgb_1/imgs_ori png 424 output/kinect/seq_1/modified_1/kinect_rgb_1/imgs_ori 10 433 0
#python3 resample_files_under_directory.py output/kinect/seq_1/ori/kinect_seg_1/imgs_ori png 424 output/kinect/seq_1/modified_1/kinect_seg_1/imgs_ori 6 414 0
#python3 resample_files_under_directory.py output/kinect/seq_2/ori/kinect_rgb_2/imgs_ori png 368 output/kinect/seq_2/modified/kinect_rgb_2/imgs_ori 4 371 0
#python3 generate_kinect_mask.py output/kinect/${seq_id}/modified/seg_${seq_id}/imgs_ori png output/kinect/${seq_id}/modified/dilated_mask_${seq_id} 0 255 0 25 
python3 generate_kinect_mask.py output/kinect/${seq_id}/modified/seg_${seq_id}/imgs_ori png output/kinect/${seq_id}/modified/rgb_dilated_mask_${seq_id}/imgs_ori 0 255 0 95 output/kinect/${seq_id}/modified/rgb_${seq_id}/imgs_ori
END
