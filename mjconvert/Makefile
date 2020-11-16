.PHONY: proto requirements install uninstall clean pytest test-cli

clean:
	rm ./mjproto/*pb2*
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

formats:
	black mjconvert
	blackdoc mjconvert
	isort mjconvert

checks:
	black --check --diff mjconvert
	blackdoc --check mjconvert
	isort --check --diff mjconvert

proto:
	python3 -m grpc_tools.protoc -I ../protos --python_out=./mjproto/ --grpc_python_out=./mjproto/ --mypy_out=./mjproto/ ../protos/mj.proto

requirements:
	python3 -m pip install -r requirements.txt

uninstall:
	python3 -m pip uninstall mjconvert -y

install: requirements proto
	python3 setup.py install

pytest: proto
	python3 -m pytest --doctest-modules

test-cli: uninstall clean install
	echo "From mjlog to mjproto"
	cat mjconvert/tests/resources/mjlog/* | mjconvert --to-mjproto --verbose | wc -l
	cat mjconvert/tests/resources/mjlog/* | mjconvert --to-mjproto-raw --verbose | wc -l
	mjconvert mjconvert/tests/resources/mjlog mjconvert/tests/resources/mjproto --to-mjproto --verbose && cat mjconvert/tests/resources/mjproto/* | wc -l
	mjconvert mjconvert/tests/resources/mjlog mjconvert/tests/resources/mjproto --to-mjproto-raw --verbose && cat mjconvert/tests/resources/mjproto/* | wc -l
	echo "From mjproto to mjlog"
	cat mjconvert/tests/resources/mjproto/* | mjconvert --to-mjlog --verbose | wc -l
	mjconvert mjconvert/tests/resources/mjproto mjconvert/tests/resources/mjlog --to-mjlog --verbose && cat mjconvert/tests/resources/mjlog/* | wc -l
	git checkout -- mjconvert/tests/resources/mjlog
