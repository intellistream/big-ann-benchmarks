FROM neurips23

RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:git-core/ppa
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format libboost-dev libboost-program-options-dev libmkl-full-dev libcpprest-dev python3.10

# 安装 OpenMP
RUN apt install -y libomp-dev

# 克隆并编译 DynaGraph
WORKDIR /home/app
COPY DynaGraph ./DynaGraph
WORKDIR /home/app/DynaGraph

# 创建构建目录并编译
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# 编译Python绑定
WORKDIR /home/app/DynaGraph/bindings
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# 安装Python绑定
RUN cd build && make install

WORKDIR /home/app
