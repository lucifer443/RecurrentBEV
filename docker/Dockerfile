FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV CUDA_HOME="/usr/local/cuda"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CPLUS_INCLUDE_PATH="/usr/local/cuda/include"
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    libgl1 \
    libglib2.0-0 \
    sshpass \
    zsh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set pypi source to aliyun
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# Open MMLab repositry versions
ARG MMENGINE_VERSION=v0.8.1
ARG MMCV_VERSION=v2.0.1
ARG MMRAZOR_VERSION=v1.0.0
ARG MMPRETRAIN_VERSION=v1.0.0rc8
ARG MMDET_VERSION=v3.1.0
ARG MMSEG_VERSION=v1.1.0
ARG MMDET3D_VERSION=v1.2.0
ARG MMDEPLOY_VERSION=v1.2.0

# Install MMEngine
RUN git clone https://github.com/open-mmlab/mmengine.git /workspace/open-mmlab-2.0/mmengine \
    && cd /workspace/open-mmlab-2.0/mmengine \
    && git checkout ${MMENGINE_VERSION} \
    && sed -i "36i\    'bev': 'bev'," mmengine/config/utils.py \
    && pip install --no-cache-dir -v -e .

# Install MMCV (cuda version)
RUN git clone https://github.com/open-mmlab/mmcv.git /workspace/open-mmlab-2.0/mmcv \
    && cd /workspace/open-mmlab-2.0/mmcv \
    && cd /workspace/open-mmlab-2.0/mmcv ${MMCV_VERSION} \
    && pip install --no-cache-dir -r requirements/optional.txt \
    && FORCE_CUDA=1 pip install --no-cache-dir -v -e .

# Install MMRazor
RUN git clone https://github.com/open-mmlab/mmrazor.git /workspace/open-mmlab-2.0/mmrazor \
    && cd /workspace/open-mmlab-2.0/mmrazor \
    && git checkout ${MMRAZOR_VERSION} \
    && pip install --no-cache-dir -v -e .

# Install MMPreTrain
RUN git clone https://github.com/open-mmlab/mmpretrain.git /workspace/open-mmlab-2.0/mmpretrain \
    && cd /workspace/open-mmlab-2.0/mmpretrain \
    && git checkout ${MMPRETRAIN_VERSION} \
    && pip install --no-cache-dir -v -e .

# Install MMDetection
RUN git clone https://github.com/open-mmlab/mmdetection.git /workspace/open-mmlab-2.0/mmdetection \
    && cd /workspace/open-mmlab-2.0/mmdetection \
    && git checkout ${MMDET_VERSION} \
    && pip install --no-cache-dir -v -e .

# Install MMSegmentation
RUN git clone https://github.com/open-mmlab/mmsegmentation.git /workspace/open-mmlab-2.0/mmsegmentation \
    && cd /workspace/open-mmlab-2.0/mmsegmentation \
    && git checkout ${MMSEG_VERSION} \
    && pip install --no-cache-dir -v -e .

# Install nuscenes-devkit-1.1.10 from source
RUN git clone https://github.com/nutonomy/nuscenes-devkit.git /workspace/open-mmlab-2.0/nuscenes-devkit \
    && cd /workspace/open-mmlab-2.0/nuscenes-devkit \
    && git checkout 9bc2f9e74d8a3dd27ef6ce83336508dd8926f867 \
    && cd setup \
    && sed -i '4d' requirements/requirements_base.txt \
    # && sed -i '1d' requirements/requirements_nuimages.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -v -e .

# Install MMDetection3D
RUN git clone https://github.com/open-mmlab/mmdetection3d.git /workspace/open-mmlab-2.0/mmdetection3d \
    && cd /workspace/open-mmlab-2.0/mmdetection3d \
    && git checkout ${MMDET3D_VERSION} \
    # && sed -i 's/scikit-image/scikit-image==0.19.3/g' requirements/runtime.txt \
    && sed -i '5d;10d' requirements/runtime.txt \
    && pip install --no-cache-dir -v -e .

# Install MMDeploy
RUN git clone https://github.com/open-mmlab/mmdeploy.git /workspace/open-mmlab-2.0/mmdeploy \
    && cd /workspace/open-mmlab-2.0/mmdeploy \
    && git checkout ${MMDEPLOY_VERSION}

# install ONNXRuntime
RUN cd /workspace/open-mmlab-2.0 \
    && pip install onnxruntime==1.14.1 \
    && wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz \
    && tar -zxvf onnxruntime-linux-x64-1.14.1.tgz \
    && rm onnxruntime-linux-x64-1.14.1.tgz
ENV ONNXRUNTIME_DIR="/workspace/open-mmlab-2.0/onnxruntime-linux-x64-1.14.1"
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:${LD_LIBRARY_PATH}

# install TensorRT
# download from https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
COPY ./TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz /workspace/open-mmlab-2.0
RUN cd /workspace/open-mmlab-2.0 \
    && tar -zxvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz \
    && rm TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz \
    && pip install TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
ENV TENSORRT_DIR="/workspace/open-mmlab-2.0/TensorRT-8.6.1.6"
ENV LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:${LD_LIBRARY_PATH}

# build ONNXRuntime and TensorRT ops
RUN cd /workspace/open-mmlab-2.0/mmdeploy \
    && mkdir -p build \
    && cd build \
    && cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} .. \
    && make -j$(nproc) \
    && make install \
    && cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} -DCUDNN_DIR=${CUDNN_DIR} .. \
    && make -j$(nproc) \
    && make install

RUN cd /workspace/open-mmlab-2.0/mmdeploy \
    && pip install --no-cache-dir -v -e .

# Install other python packages
RUN pip install --no-cache-dir -v \
    kornia \
    pycuda==2022.2.2 \
    h5py \
    tensorboard==2.11.0

# scp needs ssh locates at /usr/bin/ssh
RUN ln -s /.devenv/usr/bin/ssh /usr/bin/ssh

# Copy this DOCKERFILE
COPY ./DOCKERFILE.pytorch /workspace
