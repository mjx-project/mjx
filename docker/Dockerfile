FROM ubuntu:18.04

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
  cd /grpc/third_party/cares/cares && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j && sudo make install && sudo ldconfig && \
  cd /grpc/third_party/zlib && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j && sudo make install && sudo ldconfig && \
  cd /grpc/third_party/protobuf/cmake && mkdir build && cd build && cmake -Dprotobuf_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release .. && make -j && sudo make install && sudo ldconfig && \
  cd /grpc && \
  mkdir -p cmake/build && \
  cd cmake/build && \
  cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package -DgRPC_CARES_PROVIDER=package -DgRPC_SSL_PROVIDER=package -DCMAKE_BUILD_TYPE=Release ../..  && \
  make -j2 && \
  make install  && \
  cd / && \
  rm -rf grpc
