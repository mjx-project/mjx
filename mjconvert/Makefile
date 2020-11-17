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
	flake8 --config pyproject.toml --ignore E203,E501,W503 mjconvert
	mypy --config pyproject.toml mjconvert

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
	echo "From mjlog => mjproto"
	cat mjconvert/tests/resources/mjlog/*.mjlog | mjconvert --to-mjproto --verbose | wc -l
	cat mjconvert/tests/resources/mjlog/*.mjlog | mjconvert --to-mjproto-raw --verbose | wc -l
	mjconvert mjconvert/tests/resources/mjlog mjconvert/tests/resources/mjproto --to-mjproto --verbose && cat mjconvert/tests/resources/mjproto/* | wc -l
	rm mjconvert/tests/resources/mjproto/*.json
	mjconvert mjconvert/tests/resources/mjlog mjconvert/tests/resources/mjproto --to-mjproto-raw --verbose && cat mjconvert/tests/resources/mjproto/* | wc -l
	echo "From mjproto => mjlog_recovered"
	cat mjconvert/tests/resources/mjproto/*.json | mjconvert --to-mjlog --verbose | wc -l
	mjconvert mjconvert/tests/resources/mjproto mjconvert/tests/resources/mjlog_recovered --to-mjlog --verbose && cat mjconvert/tests/resources/mjlog_recovered/* | wc -l
	echo "Check diff"
	python3 mjconvert/diff.py mjconvert/tests/resources/mjlog mjconvert/tests/resources/mjlog_recovered
	echo "Clean up"
	rm -rf mjconvert/tests/resources/mjproto
	rm -rf mjconvert/tests/resources/mjlog_recovered
	git checkout -- mjconvert/tests/resources
