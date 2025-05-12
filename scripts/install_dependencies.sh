#!/bin/bash

set -e

mkdir -p build
cd build


if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        libboost-all-dev \
        libeigen3-dev \
        libopencv-dev \
        cuda-toolkit-11-0 \
        libcudnn8 \
        libcudnn8-dev \
        python3-dev \
        python3-pip \
        doxygen \
        graphviz
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew update
    brew install \
        cmake \
        boost \
        eigen \
        opencv \
        cuda \
        cudnn \
        python3 \
        doxygen \
        graphviz
fi

TENSORRT_VERSION="8.0.1.6"
TENSORRT_URL="https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.0.1/local_repos/nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64.deb"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    wget $TENSORRT_URL
    sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64.deb
    sudo apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.3-trt8.0.1.6-ga-20210626_1-1_amd64/nv-tensorrt-*-keyring.gpg
    sudo apt-get update
    sudo apt-get install -y tensorrt
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "TensorRT installation on macOS requires manual download from NVIDIA website"
    echo "Please visit: https://developer.nvidia.com/tensorrt"
fi

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

LIBTORCH_VERSION="1.10.0"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcu113.zip"

wget $LIBTORCH_URL
unzip libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}+cu113.zip

echo "export TENSORRT_HOME=/usr/local/tensorrt" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/tensorrt/lib" >> ~/.bashrc
echo "export LIBTORCH=/path/to/libtorch" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/path/to/libtorch/lib" >> ~/.bashrc

cmake .. \
    -DCMAKE_PREFIX_PATH=/path/to/libtorch \
    -DTENSORRT_ROOT=/usr/local/tensorrt \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 \
    -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 \
    -DBOOST_ROOT=/usr/local/opt/boost

make -j$(nproc)

echo "Dependencies installed successfully!" 