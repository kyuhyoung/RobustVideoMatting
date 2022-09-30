docker_name=u_18_cuda_11_1_1_cudnn_8_torch_1_10_2_tensorboard_2_9_1
dir_cur=${PWD##*/}
#################################################################################################
: << 'END'
#   docker build
cd docker_file/; sudo docker build --force-rm --shm-size=64g -t u_18_cuda_11_1_1_cudnn_8_torch_1_10_2_tensorboard_2_9_1 -f Dockerfile_u_18_cuda_11_1_1_cudnn_8_torch_1_10_2_tensorboard_2_9_1 .; cd -
END
#
#################################################################################################
#   docker test
#sudo docker run --rm -it --shm-size=64g --gpus '"device=1"' -v $PWD:$PWD u_18_cuda_11_1_1_cudnn_8_torch_1_10_2:latest /bin/sh -c "python3 -c 'import torch; print(\"torch.__version__ :\", torch.__version__, \"torch.version.cuda :\", torch.version.cuda)'; cd $PWD; wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth; python3 -c 'import torch; from model import MattingNetwork; model = MattingNetwork(\"mobilenetv3\").eval().cuda(); model.load_state_dict(torch.load(\"rvm_mobilenetv3.pth\"))'"
#################################################################################################
#   docker run
#sudo docker run --rm -it --shm-size=64g --gpus '"device=1"' -p 9014:9014 -e DISPLAY=$DISPLAY -v /data/matting:/data -v $PWD:/workspace/${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} fish
#sudo docker run --rm -it --shm-size=64g --gpus '"device=3"' -p 9015:9015 -e DISPLAY=$DISPLAY -v /data/matting:/data -v $PWD:/workspace/${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} fish
sudo docker run --rm -it --shm-size=64g --gpus '"device=2"' -p 9021:9021 -e DISPLAY=$DISPLAY -v /data/matting:/data -v $PWD:/workspace/${dir_cur} -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v /etc/shadow:/etc/shadow:ro -v /etc/sudoers.d:/etc/sudoers.d:ro -v /tmp/.X11-unix:/tmp/.X11-unix:rw ${docker_name} fish
