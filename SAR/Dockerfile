FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# necessary tools : python
RUN apt-get update && apt-get install -y \
    sudo \
    python3-pip \
    git \
    slurm-wlm \
    && rm -rf /var/lib/apt/lists/*


# Create a custom user with UID 1234 and GID 1234
RUN groupadd -g 1080 M2 && \
    useradd -m -u 1084 -g M2 hirano
 
# Switch to the custom user
USER hirano
 
# Set the workdir
WORKDIR /home/hirano

RUN pip install torch==1.12.0

# `requirements.txt` をコピーしてパッケージをインストール
COPY requirements.txt /home/hirano/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /home/hirano/requirements.txt
