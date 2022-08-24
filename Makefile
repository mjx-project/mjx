clean:
	rm -rf .cache
	rm -rf venv
	rm -rf examples/venv
	rm -rf cmake-build-debug
	rm -rf cmake-build-mjxdebug
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
	rm -rf tests_cpp/external/*-subbuild rm -rf .pytest_cache rm -rf __pycache__ 
	rm -rf mjx/__pycache__
	rm -rf mjx/converter/__pycache__
	rm -rf mjx/visualizer/__pycache__
	rm -rf mjxproto/__pycache__
	rm -rf tests_py/__pycache__
	rm -rf examples/__pycache__

venv:
	python3 -m venv venv
	venv/bin/python3 -m pip install --upgrade pip
	venv/bin/python3 -m pip install -r requirements.txt
	venv/bin/python3 -m pip install -r requirements-dev.txt

build: include/mjx/* include/mjx/internal/* tests_cpp/* 
	mkdir -p build && cd build && cmake .. -DMJX_BUILD_BOOST=OFF -DMJX_BUILD_GRPC=OFF -DMJX_BUILD_TESTS=ON && make -j

cpp-build: build

cpp-test: build
	./build/tests_cpp/mjx_tests_cpp

cpp-fmt:
	clang-format -i include/mjx/*.h include/mjx/*.cpp
	clang-format -i include/mjx/internal/*.h include/mjx/internal/*.cpp
	clang-format -i tests_cpp/*.cpp

dist: setup.py include/mjx/* include/mjx/internal/* mjx/* mjx/visualizer/* include/mjx/internal/mjx.proto
	which python3
	git submodule update --init
	export MJX_BUILD_BOOST=OFF && export MJX_BUILD_GRPC=OFF && python3 setup.py sdist && python3 setup.py install

py-build: dist

py-test: dist
	python3 -m pytest tests_py --import-mode=importlib 

py-fmt:
	black mjx tests_py 
	blackdoc mjx tests_py 
	isort mjx tests_py 

py-check:
	black --check --diff mjx tests_py 
	blackdoc --check mjx tests_py 
	isort --check --diff mjx tests_py 
	flake8 --config pyproject.toml --ignore E203,E501,W503 mjx # tests_py examples scripts
	# mypy --config pyproject.toml mjx

.PHONY: clean cpp-build cpp-test cpp-fmt py-fmt py-check py-test
