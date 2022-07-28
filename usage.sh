ceil_mode=false
#ceil_mode=true
for model in mobilenetv3
do
    #for checkpoint in rvm_mobilenetv3.pth
    for checkpoint in rvm_${model}
    do
        #for wh in 1920_1080 1280_720
        for wh in 1920_1080
        #for wh in 480_270
        do
            #for downsample_ratio in 1 0.5 0.25 0.125
            #for downsample_ratio in 1
            for downsample_ratio in 0.25
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
                        do                        prephix=${checkpoint}_${precision}_${device}_opset_${opset}_ceil_mode_${ceil_mode}_leavittx_220728
                            out_export=${prephix}_${wh}_downsample_${downsample_ratio}.onnx
                            out_shape_infered=${prephix}_shape_infered_downsample_${downsample_ratio}_${wh}.onnx
                            python3 export_onnx.py --model-variant ${model} --checkpoint ${checkpoint}.pth --wh ${wh} --downsample-ratio ${downsample_ratio} --precision ${precision} --opset ${opset} --device ${device} --output ${out_export} --ceil_mode ${ceil_mode}
                            python3 symbolic_shape_infer.py --input ${out_export} --output ${out_shape_infered} --precision ${precision} --verbose 3
                        done
                    done
                done
            done
        done
    done
done    
