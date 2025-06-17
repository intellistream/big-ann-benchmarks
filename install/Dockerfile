FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget gnupg2
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update

RUN apt-get install -y \
    python3-numpy \
    python3-scipy \
    python3-pip \
    build-essential \
    git \
    axel \
    wget \
    cmake \
    libboost-all-dev \
    libhdf5-dev \
    libopenblas-dev \
    libomp-dev \
    libpapi-dev \
    cuda-toolkit

RUN wget https://aka.ms/downloadazcopy-v10-linux && mv downloadazcopy-v10-linux azcopy.tgz && tar xzf azcopy.tgz --transform 's!^[^/]\+\($\|/\)!azcopy_folder\1!' 
RUN cp azcopy_folder/azcopy /usr/bin

RUN pip3 install -U pip

WORKDIR /home/app
COPY requirements_py3.10.txt run_algorithm.py ./
RUN pip3 install -r requirements_py3.10.txt

WORKDIR /home/app/CANDY-Benchmark
COPY CANDY-Benchmark/ ./
RUN pip3 install .

WORKDIR /home/app
ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
