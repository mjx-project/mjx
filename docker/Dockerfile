FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  vim \
  git \
  wget \
  cmake \
  build-essential \
  libboost-dev \
  autoconf \
  libtool \
  pkg-config

# install gRPC
RUN cd / && \
  git clone --recurse-submodules -b v1.28.1 https://github.com/grpc/grpc && \
  cd grpc && \
  mkdir -p cmake/build && \
  cd cmake/build && \
  cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/local ../.. && \
  make -j2 && \
  make install  && \
  cd / && \
  rm -rf grpc
