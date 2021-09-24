set -eu

ver=${1}
cp ../requirements*.txt ./
docker build -t sotetsuk/ubuntu-gcc-grpc:${ver} -m 6g .
rm requirements*.txt
