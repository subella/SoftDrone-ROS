# SoftDrone-Target-Pose-Estimator
Package for target pose estimating

## Requirements
Tested in Ubuntu 18.04 with [ROS Melodic](http://wiki.ros.org/melodic).

Requires [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) >10.1.

Requires [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

Requires libtorch, torchvision:
```
# Need to upgrade cmake
pip install cmake
# Refresh terminal
source ~/.bashrc 
# Need to change cu111 to match cuda version, probably
pip3 install --user torch==1.10 -f https://download.pytorch.org/whl/cu111/torch_stable.html
export TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
export CMAKE_PREFIX_PATH=$HOME/.local/lib/python3.6/site-packages/torch/
git clone --branch v0.11.1 https://github.com/pytorch/vision/
mkdir vision/build && cd vision/build && \
	cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=on -DTORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST && \
	make -j && make install
```

Requires [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus)


## Building:
```
export Torch_DIR=$HOME/.local/lib/python3.6/site-packages/torch/
catkin build
```
