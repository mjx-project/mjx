clean:
	rm -rf venv
	rm -rf examples/venv
	rm -rf cmake-build-debug
	rm -rf build
	rm -rf docker-build
	rm -rf dist
	rm -rf include/mjx/internal/mjx.grpc.pb.cc
	rm -rf include/mjx/internal/mjx.grpc.pb.h
	rm -rf include/mjx/internal/mjx.pb.cc
	rm -rf include/mjx/internal/mjx.pb.h
	rm -rf mjx.egg-info
	rm -rf external/*-build
	rm -rf external/*-subbuild
	rm -rf tests_cpp/external/*-build
	rm -rf tests_cpp/external/*-subbuild
	rm -rf .pytest_cache
	rm -rf __pycache__ 
	rm -rf mjx/__pycache__
	rm -rf mjx/converter/__pycache__
	rm -rf mjx/visualizer/__pycache__
	rm -rf mjxproto/__pycache__
	rm -rf tests_py/__pycache__
	rm -rf examples/__pycache__

venv:
	python3 -m venv venv

build: include/mjx/* include/mjx/internal/* tests_cpp/*
	mkdir -p build && cd build && cmake .. -DMJX_BUILD_BOOST=OFF -DMJX_BUILD_GRPC=OFF -DMJX_BUILD_TESTS=ON && make -j

cpptest: build
	./build/tests_cpp/mjx_tests_cpp

cppfmt:
	clang-format -i include/mjx/*.h include/mjx/*.cpp
	clang-format -i include/mjx/internal/*.h include/mjx/internal/*.cpp
	clang-format -i tests_cpp/*.cpp
	clang-format -i scripts/*.cpp

dist: setup.py include/mjx/* include/mjx/internal/* mjx/* mjx/converter/* mjx/visualizer/* include/mjx/internal/mjx.proto
	which python3
	git submodule update --init
	export MJX_BUILD_BOOST=OFF && export MJX_BUILD_GRPC=OFF && python3 setup.py sdist && python3 setup.py install

pyfmt:
	black mjx tests_py examples scripts
	blackdoc mjx tests_py examples scripts
	isort mjx tests_py examples scripts

pycheck:
	black --check --diff mjx tests_py examples scripts
	blackdoc --check mjx tests_py examples scripts
	isort --check --diff mjx tests_py 
	flake8 --config pyproject.toml --ignore E203,E501,W503 mjx # tests_py examples scripts
	# mypy --config pyproject.toml mjx

pytest: dist
	python3 -m pytest tests_py --import-mode=importlib 

docker-build:
	docker run -it -v ${CURDIR}:/mahjong sotetsuk/ubuntu-gcc-grpc:latest  /bin/bash -c "cd /mahjong && mkdir -p docker-build && cd docker-build && cmake .. && make -j"

docker-test: docker-build
	docker run -it -v ${CURDIR}:/mahjong sotetsuk/ubuntu-gcc-grpc:latest  /bin/bash -c "/mahjong/docker-build/tests_cpp/mjx_tests_cpp"

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


.PHONY: clean cpptest cppfmt pyfmt pycheck pytest docker-test docker-all docker-clion-stop docker-clion-start docker-plantuml-start docker-plantuml-stop
