set -eu

ver=${1}
docker build -t sotetsuk/ubuntu-gcc-grpc:${ver} -m 6g .
