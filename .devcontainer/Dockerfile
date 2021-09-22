FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  vim \
  git \
  wget \
  curl \
  cmake \
  build-essential \
  libboost-dev \
  autoconf \
  libtool \
  pkg-config \
  unzip \
  git-lfs

# install gRPC
RUN cd / && \
  git clone --recurse-submodules -b v1.39.0 https://github.com/grpc/grpc && \
  cd grpc && \
  mkdir -p cmake/build && \
  cd cmake/build && \
  cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
      ../.. && \
  make -j 2 && \
  make install && \
  cd ../.. && \
  mkdir -p third_party/abseil-cpp/cmake/build && \
  cd third_party/abseil-cpp/cmake/build && \
  cmake -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
      -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE \
      ../.. && \
  make -j 2 && \
  make install && \
  rm -rf /grpc

# install python
ARG PY_VERSION=3.9
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=${PY_VERSION} && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
