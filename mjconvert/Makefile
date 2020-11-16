.PHONY: proto install uninstall clean pytest test-cli test-cli-mjlog-to-proto test-cli-mjgproto-to-mjlog

clean:
	rm ./mjconvert/mj_pb2.py ./mjconvert/mj_pb2_grpc.py
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

formats:
	ls mjconvert/*.py | grep -v pb2 | xargs black
	ls mjconvert/*.py | grep -v pb2 | xargs blackdoc
	ls mjconvert/*.py | grep -v pb2 | xargs isort

checks:
	ls mjconvert/*.py | grep -v pb2 | xargs black --check --diff
	ls mjconvert/*.py | grep -v pb2 | xargs blackdoc --check
	ls mjconvert/*.py | grep -v pb2 | xargs isort --check --diff
	ls mjconvert/*.py | grep -v mjproto | xargs mypy --config pyproject.toml 

proto:
	python3 -m grpc_tools.protoc -I ../protos --python_betterproto_out=./mjconvert/ ../protos/mj.proto

uninstall:
	python3 -m pip uninstall mjconvert -y

install: proto
	python3 -m pip install -r requirements.txt
	python3 setup.py install

pytest: proto
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
