set -eu

ver=${1}
cp ../mjconvert/requirements.txt ./mjconvert_requirements.txt
docker build -t sotetsuk/ubuntu-gcc-grpc:${ver} -m 6g .
docker build -t sotetsuk/ubuntu-gcc-grpc:latest -m 6g .
rm ./mjconvert_requirements.txt
