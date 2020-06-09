set -eu

ver=${1}
docker build -t mj:${ver} -m 6g .
