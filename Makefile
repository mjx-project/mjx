clean:
	cd mjconvert && make clean
	rm -rf cmake-build-debug || "cmake-build-debug is already deleted"
	rm -rf build || "debug is already deleted"
	rm -rf docker-build || "docker-debug is already deleted"
	rm -rf mjx/*pb* || "mjx/*pb* is already deleted"
	rm -rf mjx/external_libs/* || "mjx/external_libs/** is already deleted"

build: mjx tests mjx.proto
	git submodule update --init
	mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DgRPC_BUILD_TESTS=OFF -DgRPC_SSL_PROVIDER=package -DGRPC_FETCHCONTENT=ON -DFETCHCONTENT_SOURCE_DIR_GRPC="$(pwd)" .. && make -j

test: build
	./build/tests/mjx_test

all: clean test

fmt:
	clang-format -i *.cpp
	clang-format -i mjx/*.h mjx/*.cpp
	clang-format -i tests/*.cpp
	clang-format -i scripts/*.cpp

docker-build:
	docker run -it -v ${CURDIR}:/mahjong sotetsuk/ubuntu-gcc-grpc:latest  /bin/bash -c "cd /mahjong && mkdir -p docker-build && cd docker-build && cmake .. && make -j"

docker-test: docker-build
	docker run -it -v ${CURDIR}:/mahjong sotetsuk/ubuntu-gcc-grpc:latest  /bin/bash -c "/mahjong/docker-build/tests/mjx_test"

docker-all: clean docker-test

docker-clion-start: docker-clion-stop
	docker run -d --cap-add sys_ptrace -p 127.0.0.1:2222:22 --name mahjong-remote-clion sotetsuk/ubuntu-gcc-grpc-clion:latest
	ssh-keygen -f "${HOME}/.ssh/known_hosts" -R "[localhost]:2222"

docker-clion-stop:
	docker rm -f mahjong-remote-clion || true

docker-plantuml-start: docker-plantuml-stop
	docker run -d -p 8080:8080 --name mahjong-plantuml plantuml/plantuml-server:jetty

docker-plantuml-stop:
	docker rm -f mahjong-plantuml || true


.PHONY: clean test all fmt docker-test docker-all docker-clion-stop docker-clion-start docker-plantuml-start docker-plantuml-stop
