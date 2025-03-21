FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    gnutls-bin \
    openssh-client \
    libghc-x11-dev \
    gcc-multilib \
    g++-multilib \
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    xvfb \
    mesa-utils \
    libegl1-mesa \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    unzip \
    openjdk-8-jdk 

RUN python -m pip uninstall opencv -y &&\
    python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple &&\
    python -m pip install minestudio==1.1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple &&\
    python -m pip install opencv-python==4.8.0.74 opencv-python-headless==4.8.0.74 -i https://pypi.tuna.tsinghua.edu.cn/simple

USER root

# ARG http_proxy=http://172.17.40.11:7890
# ARG https_proxy=http://172.17.40.11:7890
ARG HF_ENDPOINT="https://hf-mirror.com"

WORKDIR /app
RUN mkdir /.cache && mkdir /app/minestudio_temp_dir 
ENV MINESTUDIO_TEMP_DIR="/app/minestudio_temp_dir"
RUN python -m minestudio.simulator.entry -y

RUN git clone https://github.com/CraftJarvis/MineStudio.git &&\
    cd MineStudio/minestudio/utils/realtime_sam &&\
    python -m pip install --no-build-isolation -e . -i https://pypi.tuna.tsinghua.edu.cn/simple &&\
    cd checkpoints &&\
    bash download_ckpts.sh

RUN python -m pip install gradio==5.9.1 pillow==11.0.0 &&\
    git clone https://github.com/CraftJarvis/ROCKET-2.git &&\
    cd ROCKET-2 &&\
    python model.py

WORKDIR /app/ROCKET-2

CMD ["python", "launch.py", "--env-conf", "./env_conf", "--sam-path", "/app/MineStudio/minestudio/utils/realtime_sam/checkpoints"]