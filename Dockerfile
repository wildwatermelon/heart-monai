#FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   nano \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    torch

WORKDIR /workspace
COPY . heart-monai/
RUN cd heart-monai/

#    && \
#    python3 -m pip install --no-cache-dir .
# CMD ["/bin/bash"]