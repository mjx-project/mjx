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
	venv/bin/python3 -m pip install --upgrade pip
	venv/bin/python3 -m pip install -r requirements.txt
	venv/bin/python3 -m pip install -r requirements-dev.txt

build: include/mjx/* include/mjx/internal/* tests_cpp/* scripts/*
	mkdir -p build && cd build && cmake .. -DMJX_BUILD_BOOST=OFF -DMJX_BUILD_GRPC=OFF -DMJX_BUILD_TESTS=ON && make -j

cpp-build: build

cpp-test: build
	./build/tests_cpp/mjx_tests_cpp

cpp-fmt:
	clang-format -i include/mjx/*.h include/mjx/*.cpp
	clang-format -i include/mjx/internal/*.h include/mjx/internal/*.cpp
	clang-format -i tests_cpp/*.cpp
	clang-format -i scripts/*.cpp

dist: setup.py include/mjx/* include/mjx/internal/* mjx/* mjx/converter/* mjx/visualizer/* include/mjx/internal/mjx.proto
	which python3
	git submodule update --init
	export MJX_BUILD_BOOST=OFF && export MJX_BUILD_GRPC=OFF && python3 setup.py sdist && python3 setup.py install

py-build: dist

py-test: dist
	python3 -m pytest tests_py --import-mode=importlib 

py-fmt:
	black mjx tests_py examples scripts
	blackdoc mjx tests_py examples scripts
	isort mjx tests_py examples scripts

py-check:
	black --check --diff mjx tests_py examples scripts
	blackdoc --check mjx tests_py examples scripts
	isort --check --diff mjx tests_py 
	flake8 --config pyproject.toml --ignore E203,E501,W503 mjx # tests_py examples scripts
	# mypy --config pyproject.toml mjx

cli-test:
	echo "From mjlog => mjxproto"
	cat tests_cpp/resources/mjlog/*.mjlog | mjx convert --to-mjxproto --verbose | wc -l
	cat tests_cpp/resources/mjlog/*.mjlog | mjx convert --to-mjxproto --compress --verbose | wc -l
	cat tests_cpp/resources/mjlog/*.mjlog | mjx convert --to-mjxproto-raw --verbose | wc -l
	mkdir -p tests_cpp/resources/mjxproto
	mjx convert tests_cpp/resources/mjlog tests_cpp/resources/mjxproto --to-mjxproto --verbose && cat tests_cpp/resources/mjxproto/* | wc -l
	rm -rf tests_cpp/resources/mjxproto/*.json
	mjx convert tests_cpp/resources/mjlog tests_cpp/resources/mjxproto --to-mjxproto --compress --verbose && cat tests_cpp/resources/mjxproto/* | wc -l
	rm tests_cpp/resources/mjxproto/*.json
	mjx convert tests_cpp/resources/mjlog tests_cpp/resources/mjxproto --to-mjxproto-raw --verbose && cat tests_cpp/resources/mjxproto/* | wc -l
	echo "From mjxproto => mjlog_recovered"
	cat tests_cpp/resources/mjxproto/*.json | mjx convert --to-mjlog --verbose | wc -l
	mkdir -p tests_cpp/resources/mjlog_recovered
	mjx convert tests_cpp/resources/mjxproto tests_cpp/resources/mjlog_recovered --to-mjlog --verbose && cat tests_cpp/resources/mjlog_recovered/* | wc -l
	echo "Check diff"
	python3 mjx/utils/diff.py tests_cpp/resources/mjlog tests_cpp/resources/mjlog_recovered
	echo "Clean up"
	rm -rf tests_cpp/resources/mjxproto
	rm -rf tests_cpp/resources/mjlog_recovered
	git checkout -- tests_cpp/resources


.PHONY: clean cpp-build cpp-test cpp-fmt py-fmt py-check py-test cli-test
