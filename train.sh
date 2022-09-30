: << 'END'
#################################################################################################
#   train stage 1 
#   Author's setting. 
#python3 train.py --model-variant mobilenetv3 --dataset videomatte --resolution-lr 512 --seq-length-lr 15 --learning-rate-backbone 0.0001 --learning-rate-aspp 0.0002 --learning-rate-decoder 0.0002 --learning-rate-refiner 0 --checkpoint-dir checkpoint/stage1 --log-dir log/stage1 --epoch-start 0 --epoch-end 20 --is_hair 
#   For k-hairstyle stage 1 
python3 train.py --model-variant mobilenetv3 --dataset videomatte --resolution-lr 512 --seq-length-lr 15 --learning-rate-backbone 0.00001 --learning-rate-aspp 0.00002 --learning-rate-decoder 0.00002 --learning-rate-refiner 0 --checkpoint-dir checkpoint/stage1 --log-dir log/stage1 --epoch-start 0 --epoch-end 20 --is_hair 
END

: << 'END'
##################################################################################################
#   train stage 2 
#   Author's setting.
#python train.py --model-variant mobilenetv3 --dataset videomatte --resolution-lr 512 --seq-length-lr 50 --learning-rate-backbone 0.00005 --learning-rate-aspp 0.0001 --learning-rate-decoder 0.0001 --learning-rate-refiner 0 --checkpoint checkpoint/stage1/epoch-19.pth --checkpoint-dir checkpoint/stage2 --log-dir log/stage2 --epoch-start 20 --epoch-end 22
#   For k-hairstyle stage 2
python train.py --model-variant mobilenetv3 --dataset videomatte --resolution-lr 512 --seq-length-lr 50 --learning-rate-backbone 0.000005 --learning-rate-aspp 0.00001 --learning-rate-decoder 0.00001 --learning-rate-refiner 0 --checkpoint checkpoint/stage1/ep_06_ib_04798_val_loss_0.08751676173903891.pth --checkpoint-dir checkpoint/stage2 --log-dir log/stage2 --epoch-start 7 --epoch-end 9 --checkpoint-save-interval 2000  --is_hair
END

#: << 'END'
##################################################################################################
#   train stage 3 
#   Author's setting.
#python train.py --model-variant mobilenetv3 --dataset videomatte --train-hr --resolution-lr 512 --resolution-hr 2048 --seq-length-lr 40 --seq-length-hr 6 --learning-rate-backbone 0.00001 --learning-rate-aspp 0.00001 --learning-rate-decoder 0.00001 --learning-rate-refiner 0.0002 --checkpoint checkpoint/stage2/epoch-21.pth --checkpoint-dir checkpoint/stage3 --log-dir log/stage3 --epoch-start 22 --epoch-end 23
#   For k-hairstyle 0001.hqset stage 3
python train.py --model-variant mobilenetv3 --dataset videomatte --train-hr --resolution-lr 512 --resolution-hr 1024 --seq-length-lr 40 --seq-length-hr 6 --learning-rate-backbone 0.000001 --learning-rate-aspp 0.000001 --learning-rate-decoder 0.000001 --learning-rate-refiner 0.00002 --checkpoint checkpoint/stage2/ep_08_ib_00336_val_loss_0.08507319947178603.pth --checkpoint-dir checkpoint/stage3 --log-dir log/stage3 --epoch-start 9 --epoch-end 10 --checkpoint-save-interval 2000 --is_hair
#END
