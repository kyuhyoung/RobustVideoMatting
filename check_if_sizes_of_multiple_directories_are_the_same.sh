#dir_1=/data/matting/k-hairstyle/Training/0002.mqset
#ir_1=/data/matting/k-hairstyle/Validation/0002.mqset
#dir_1=/data/k-hairstyle/Validation/0003.rawset
#dir_1=/data/k-hairstyle/Validation/0003.rawset_modified_shit
dir_1=/data/k-hairstyle/Training/0003.rawset
#dir_1=/data/k-hairstyle/Training/0003.rawset_modified_shit_7
command_1="find ${dir_1} -mindepth 1 -maxdepth 1 -type d"
str_zero="0"
for d_hair_style in `${command_1}`
do  
    #echo "d_hair_style : ${d_hair_style}"
    command_2="find ${d_hair_style} -mindepth 1 -maxdepth 1 -type d"
    for d_id in `${command_2}`
    do
        #echo "d_id : ${d_id}"
        str_n_json=`find ${d_id} -mindepth 1 -maxdepth 1 -name '*.json' | wc -l`
        #str_n_png=`find ${d_id}/rvm_comp -mindepth 1 -maxdepth 1 -name '*.png' | wc -l`
        str_n_png=`find ${d_id}/rvm_alpha -mindepth 1 -maxdepth 1 -name '*.png' | wc -l`
        str_n_jpg=`find ${d_id} -mindepth 1 -maxdepth 1 -name '*.jp*g' | wc -l`
        str_n_JPG=`find ${d_id} -mindepth 1 -maxdepth 1 -name '*.JP*G' | wc -l`
        str_n_hair=`find ${d_id}/hair_seg_map -mindepth 1 -maxdepth 1 -name '*.png' | wc -l`
        
        if [ "$str_n_png" != "$str_zero" ]; then
            if [ "$str_n_json" != "$str_n_png" ]; then
                echo "# of png is not the same as # of json at ${d_id}"
                echo "str_n_json : ${str_n_json}"
                echo "str_n_png : ${str_n_png}"
                echo "str_n_jpg : ${str_n_jpg}"
                echo "str_n_JPG : ${str_n_JPG}"
                echo "str_n_hair : ${str_n_hair}"
                echo ""
            fi
            if [ "$str_n_json" != "$str_n_hair" ]; then
                echo "# of hair is not the same as # of json at ${d_id}"
                echo "str_n_json : ${str_n_json}"
                echo "str_n_png : ${str_n_png}"
                echo "str_n_jpg : ${str_n_jpg}"
                echo "str_n_JPG : ${str_n_JPG}"
                echo "str_n_hair : ${str_n_hair}"
                echo ""
            fi
: << 'END'
            if [ "$str_n_jpg" != "$str_n_png" ] && [ "$str_n_JPG" != "$str_n_png" ]; then
                echo "# of jpg or JPG is not the same as # of png at ${d_id}"
                echo "str_n_json : ${str_n_json}"
                echo "str_n_png : ${str_n_png}"
                echo "str_n_jpg : ${str_n_jpg}"
                echo "str_n_JPG : ${str_n_JPG}"
                echo "str_n_hair : ${str_n_hair}"
                echo ""
            fi
END
        else
            echo "# of png is zero at ${d_id}"
            echo ""
        fi
        #exit
    done
    #exit
done    
