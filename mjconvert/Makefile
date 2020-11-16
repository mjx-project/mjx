.PHONY: proto install uninstall clean pytest test-cli

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
	ls mjconvert/*.py | grep -v pb2 | xargs flake8 --config pyproject.toml --ignore E501,E203,W503
	ls mjconvert/*.py | grep -v pb2 | xargs isort --check --diff

proto:
	python3 -m grpc_tools.protoc -I ../protos --python_out=./mjconvert/ --grpc_python_out=./mjconvert/ ../protos/mj.proto

uninstall:
	python3 -m pip uninstall mjconvert -y

install: proto
	python3 -m pip install -r requirements.txt
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
