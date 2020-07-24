set -eu

ver=${1}
docker build -t sotetsuk/ubuntu-gcc-grpc:${ver} -m 6g .
docker build -t sotetsuk/ubuntu-gcc-grpc:latest -m 6g .
