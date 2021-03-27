FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  vim \
  git \
  wget \
  curl \
  cmake \
  build-essential \
  autoconf \
  libtool \
  pkg-config \
  unzip \
  git-lfs

# install python
ARG PY_VERSION=3.8
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=${PY_VERSION} && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# install mjconvert dependency via pip
COPY ./mjconvert_requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
