#################################################################################################
#: << 'END'
#
#   train 
#
#   original training. 
#
#python3 train.py --model-variant mobilenetv3 --dataset videomatte --resolution-lr 512 --seq-length-lr 15 --learning-rate-backbone 0.0001 --learning-rate-aspp 0.0002 --learning-rate-decoder 0.0002 --learning-rate-refiner 0 --checkpoint-dir checkpoint/stage1 --log-dir log/stage1 --epoch-start 0 --epoch-end 20 --is_hair 
python3 train.py --model-variant mobilenetv3 --dataset videomatte --resolution-lr 512 --seq-length-lr 15 --learning-rate-backbone 0.00001 --learning-rate-aspp 0.00002 --learning-rate-decoder 0.00002 --learning-rate-refiner 0 --checkpoint-dir checkpoint/stage1 --log-dir log/stage1 --epoch-start 0 --epoch-end 20 --is_hair 
#python3 train.py --model-variant mobilenetv3 --dataset videomatte --resolution-lr 512 --seq-length-lr 15 --learning-rate-backbone 0.0001 --learning-rate-aspp 0.0002 --learning-rate-decoder 0.0002 --learning-rate-refiner 0 --checkpoint-dir checkpoint/stage1 --log-dir log/stage1 --epoch-start 0 --epoch-end 20 --is_hair --checkpoint-save-interval 10 --disable-validation
#python3 train.py --model-variant mobilenetv3 --dataset videomatte --resolution-lr 512 --seq-length-lr 15 --learning-rate-backbone 0.0001 --learning-rate-aspp 0.0002 --learning-rate-decoder 0.0002 --learning-rate-refiner 0 --checkpoint-dir checkpoint/stage1 --log-dir log/stage1 --epoch-start 0 --epoch-end 20 --checkpoint-save-interval 10 --disable-validation  
#END
#
