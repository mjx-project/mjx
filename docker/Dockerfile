FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  vim \
  git \
  wget \
  cmake \
  build-essential \
  autoconf \
  libtool \
  pkg-config \
  lsb-release \
  software-properties-common

# install llvm following https://apt.llvm.org/
# clang++-10 -stdlib=libc++ -std=c++17 -v test.cc
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" && \
  apt-get install -y clang-format clang-tidy clang-tools clang clangd libc++-dev libc++1 libc++abi-dev libc++abi1 libclang-dev libclang1 liblldb-dev libllvm-ocaml-dev libomp-dev libomp5 lld lldb llvm-dev llvm-runtime llvm python-clang

# install gRPC
RUN cd / && \
  git clone --recurse-submodules -b v1.28.1 https://github.com/grpc/grpc && \
  cd grpc && \
  mkdir -p cmake/build && \
  cd cmake/build && \
  export CC=/usr/bin/clang-10 && export CXX=/usr/bin/clang++-10 && export CXXFLAGS=-stdlib=libc++ && \
  cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/local ../.. && \
  make -j2 && \
  make install  && \
  cd / && \
  rm -rf grpc
