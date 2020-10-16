clean:
	rm tenhou/*pb2.py tenhou/*pb2_grpc.py
	rm mj/*.pb.cc mj/*.pb.h
	rm -rf build
	rm -rf docker-build

build: 
	mkdir -p build && cd build && cmake .. && make -j

test: build
	./build/test/mj_test

all: clean test

docker-build:
	docker run -it -v ${CURDIR}:/mahjong sotetsuk/ubuntu-gcc-grpc:latest  /bin/bash -c "cd /mahjong && mkdir -p docker-build && cd docker-build && cmake .. && make -j"

docker-test: docker-build
	docker run -it -v ${CURDIR}:/mahjong sotetsuk/ubuntu-gcc-grpc:latest  /bin/bash -c "/mahjong/docker-build/test/mj_test"

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


.PHONY: clean test all docker-test docker-all docker-clion-stop docker-clion-start docker-plantuml-start docker-plantuml-stop 
