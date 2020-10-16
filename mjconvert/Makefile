.PHONY: proto install uninstall clean test-cli

clean:
	rm ./src/mj_pb2.py ./src/mj_pb2_grpc.py
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

proto:
	python -m grpc_tools.protoc -I ../protos --python_out=./src/ --grpc_python_out=./src/ ../protos/mj.proto

uninstall:
	pip uninstall mjconvert -y

install: proto
	python setup.py install 

test-cli: install
	cat resources/mjlog/* | mjconvert --to-mjproto --verbose | wc -l
	cat resources/mjlog/* | mjconvert --to-mjproto-raw --verbose | wc -l
	mjconvert resources/mjlog resources/mjproto --to-mjproto --verbose && cat resources/mjproto/* | wc -l
	mjconvert resources/mjlog resources/mjproto --to-mjproto-raw --verbose && cat resources/mjproto/* | wc -l
