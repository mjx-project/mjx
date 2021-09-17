set -eu

ver=${1}
cp ../pymjx/requirements.txt ./mjx_requirements.txt
docker build -t sotetsuk/ubuntu-gcc-grpc:${ver} -m 6g .
docker build -t sotetsuk/ubuntu-gcc-grpc:latest -m 6g .
rm ./mjx_requirements.txt
