.PHONY: proto install uninstall clean test-cli test-cli-mjlog-to-proto test-cli-mjgproto-to-mjlog

clean:
	rm ./mjconvert/mj_pb2.py ./mjconvert/mj_pb2_grpc.py
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

proto:
	python3 -m grpc_tools.protoc -I ../protos --python_out=./mjconvert/ --grpc_python_out=./mjconvert/ ../protos/mj.proto

uninstall:
	python3 -m pip uninstall mjconvert -y

install: proto
	python3 -m pip install -r requirements.txt
	python3 setup.py install

pytest:
	python3 -m pytest --doctest-modules

test-cli: uninstall clean install test-cli-mjlog-to-proto test-cli-mjproto-to-mjlog

test-cli-mjlog-to-proto:
	cat resources/mjlog/* | mjconvert --to-mjproto --verbose | wc -l
	cat resources/mjlog/* | mjconvert --to-mjproto-raw --verbose | wc -l
	mjconvert resources/mjlog resources/mjproto --to-mjproto --verbose && cat resources/mjproto/* | wc -l
	mjconvert resources/mjlog resources/mjproto --to-mjproto-raw --verbose && cat resources/mjproto/* | wc -l
	git checkout -- resources/mjproto

test-cli-mjproto-to-mjlog:
	cat resources/mjproto/* | mjconvert --to-mjlog --verbose | wc -l
	mjconvert resources/mjproto resources/mjlog --to-mjlog --verbose && cat resources/mjlog/* | wc -l
	git checkout -- resources/mjlog
