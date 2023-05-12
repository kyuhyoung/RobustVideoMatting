#   ceil_mode should be false sicne current Barracuda does NOT support AvgPool2D(ceil_mode = True)
ceil_mode=false
for model in mobilenetv3
#for model in resnet50
#for model in resnet50 mobilenetv3
do
    #for checkpoint in rvm_mobilenetv3.pth
    for checkpoint in rvm_${model}
    do
        #for wh in 2560_1440 1440_2560 1920_1080 1080_1920
        #for wh in 1920_1080 1280_720
        #for wh in 1920_1080
        #for wh in 1920_1080 960_540
        #for wh in 960_540
        #for wh in 480_270
        for wh in 1080_1920
        #for wh in 1080_1920 2160_3840
        #for wh in 2160_3840
        #for wh in 1920_1088 1088_1920
        #for wh in 1920_1088 1088_1920 1080_1920
        #for wh in 720_1280
        #for wh in 6000_4000 4000_6000
        #for wh in 6000_4000
        do
            #for downsample_ratio in 0.6 0.7 0.8 0.9
            #for downsample_ratio in 1 0.5 0.25 0.125
            #for downsample_ratio in 1 0.25
            for downsample_ratio in 0.25
            #for downsample_ratio in 1
            #for downsample_ratio in 0.25 0.125
            #for downsample_ratio in 0.25 0.5
            #for downsample_ratio in 0.3 0.35 0.4
            #for downsample_ratio in 0.125
            do
                #for precision in float32 float16
                #for precision in float32
                for precision in float16
                do
                    #for device in cuda cpu
                    for device in cuda
                    do
                        #for opset in 12 9
                        for opset in 11
                        do                        
                            prephix=${checkpoint}_ceil_mode_${ceil_mode}_${device}_opset_${opset}_${precision}
                            out_export=${prephix}_downsample_${downsample_ratio}_${wh}.onnx
                            python3 export_onnx.py --model-variant ${model} --checkpoint ${checkpoint}.pth --wh ${wh} --downsample-ratio ${downsample_ratio} --precision ${precision} --opset ${opset} --device ${device} --output ${out_export} --ceil_mode ${ceil_mode} --no_verbose
                            #out_shape_infered=${prephix}_shape_infered_downsample_${downsample_ratio}_${wh}.onnx
                            #python3 symbolic_shape_infer.py --input ${out_export} --output ${out_shape_infered} --precision ${precision} --verbose 3
                        done
                    done
                done
            done
        done
    done
done    
