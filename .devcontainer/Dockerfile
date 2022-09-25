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
  git-lfs \
  python3-pip \
  python3-venv

# install gRPC
RUN cd / && \
  git clone --recurse-submodules -b v1.49.1 https://github.com/grpc/grpc && \
  cd grpc && \
  mkdir -p cmake/build && \
  cd cmake/build && \
  cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      ../.. && \
  make -j 4 && \
  make install && \
  cd ../.. && \
  mkdir -p third_party/abseil-cpp/cmake/build && \
  cd third_party/abseil-cpp/cmake/build && \
  cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE \
      ../.. && \
  make -j 4 && \
  make install && \
  rm -rf /grpc

# install googletest
RUN cd /tmp && \
  git clone -b release-1.11.0 https://github.com/google/googletest.git && \
  cd googletest && \
  mkdir build && \
  cd build && \
  cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local && \
  make -j 4 && \
  make install && \
  ldconfig && \
  rm -rf /tmp/googletest
