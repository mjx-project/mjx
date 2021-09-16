ARG VARIANT=ubuntu-20.04
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  vim \
  git \
  curl \
  libboost-dev \
  autoconf \
  libtool \
  pkg-config

# install gRPC
# NOTE: take care grpc version!
RUN cd / && \
  git clone --recurse-submodules -b v1.39.0 https://github.com/grpc/grpc && \
  cd grpc && \
  mkdir -p cmake/build && \
  cd cmake/build && \
  cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/local ../.. && \
  make -j 4 && \
  make install  && \
  cd / && \
  rm -rf grpc

# install python
ARG PY_VERSION=3.9
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=${PY_VERSION} && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
