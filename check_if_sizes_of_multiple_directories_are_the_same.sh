#dir_1=/data/matting/k-hairstyle/Training/0002.mqset
dir_1=/data/matting/k-hairstyle/Validation/0002.mqset
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
        str_n_png=`find ${d_id}/rvm_comp -mindepth 1 -maxdepth 1 -name '*.png' | wc -l`
        
        #echo "str_n_json : ${str_n_json}"
        #echo "str_n_png : ${str_n_png}"
        if [ "$str_n_png" != "$str_zero" ]; then
            if [ "$str_n_json" != "$str_n_png" ]; then
                echo "# of png is not the same as # of json at ${d_id}"
                echo ""
            fi
        else
            echo "# of png is zero at ${d_id}"
            echo ""
        fi
        #exit
    done
    #exit
done    
