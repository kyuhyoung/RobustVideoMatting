#: << 'END'
#################################################################################################
docker_name=u_18_cuda_11_1_1_cudnn_8_torch_1_10_2_tensorboard_2_9_1
#   docker build
cd docker_file/; sudo docker build --force-rm --shm-size=64g -t ${docker_name} -f Dockerfile_${docker_name} .; cd -
#END
#
#################################################################################################
#   docker test
#sudo docker run --rm -it --shm-size=64g --gpus '"device=1"' -v $PWD:$PWD u_18_cuda_11_1_1_cudnn_8_torch_1_10_2:latest /bin/sh -c "python3 -c 'import torch; print(\"torch.__version__ :\", torch.__version__, \"torch.version.cuda :\", torch.version.cuda)'; cd $PWD; wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth; python3 -c 'import torch; from model import MattingNetwork; model = MattingNetwork(\"mobilenetv3\").eval().cuda(); model.load_state_dict(torch.load(\"rvm_mobilenetv3.pth\"))'"
#################################################################################################
#   docker run
dir_cur=/workspace/${PWD##*/} 
dir_data=/data/matting
sudo docker run --rm -it --shm-size=64g --gpus '"device=1"' -p 9010:9010 -e DISPLAY=$DISPLAY -w ${dir_cur} -v ${dir_data}:${dir_data} -v $PWD:${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} fish
