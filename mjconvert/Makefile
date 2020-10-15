.PHONY: proto install uninstall clean

clean:
	rm ./src/mj_pb2.py ./src/mj_pb2_grpc.py
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

proto:
	python3 -m grpc_tools.protoc -I ../protos --python_out=./src/ --grpc_python_out=./src/ ../protos/mj.proto

uninstall:
	pip3 uninstall mjconvert -y

install: uninstall clean proto
	python3 setup.py install 

