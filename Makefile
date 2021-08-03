clean:
	cd pymjx && make clean
	rm -rf cmake-build-debug
	rm -rf build
	rm -rf docker-build
	rm -rf mjx/include/mjx/internal/*pb*
	rm -rf dist
	rm -rf venv

venv:
	python3 -m venv venv

build: mjx tests
	mkdir -p build && cd build && cmake .. -DMJX_USE_SYSTEM_BOOST=ON -DMJX_USE_SYSTEM_GRPC=ON -DMJX_BUILD_TESTS=ON && make -j

test: build
	./build/tests/mjx_test

fmt:
	clang-format -i mjx/include/mjx/internal/*.h mjx/include/mjx/internal/*.cpp
	clang-format -i tests/*.cpp
	clang-format -i scripts/*.cpp

dist: setup.py mjx pymjx
	which python3
	export MJX_USE_SYSTEM_BOOST=ON && export MJX_USE_SYSTEM_GRPC=ON && python3 setup.py sdist && python3 setup.py install

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


.PHONY: clean test fmt docker-test docker-all docker-clion-stop docker-clion-start docker-plantuml-start docker-plantuml-stop
