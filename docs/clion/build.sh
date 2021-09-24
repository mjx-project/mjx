set -eu

ver=${1}
docker build -t sotetsuk/ubuntu-gcc-grpc-clion:${ver} .
docker build -t sotetsuk/ubuntu-gcc-grpc-clion:latest .
