.PHONY: proto requirements install uninstall clean pytest test-cli

clean:
	rm ./mjxproto/*pb2* || echo "Already deleted"
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
	python3 -m grpc_tools.protoc -I ../ --python_out=./mjxproto/ --grpc_python_out=./mjxproto/ --mypy_out=./mjxproto/ ../mjx/internal/mjx.proto

requirements:
	python3 -m pip install -r requirements.txt

uninstall:
	python3 -m pip uninstall mjconvert -y

install: requirements proto
	python3 setup.py install

pytest: proto
	python3 -m pytest --doctest-modules

test-cli: uninstall clean install
	echo "From mjlog => mjxproto"
	cat mjconvert/tests/resources/mjlog/*.mjlog | mjconvert --to-mjxproto --verbose | wc -l
	cat mjconvert/tests/resources/mjlog/*.mjlog | mjconvert --to-mjxproto --compress --verbose | wc -l
	cat mjconvert/tests/resources/mjlog/*.mjlog | mjconvert --to-mjxproto-raw --verbose | wc -l
	mjconvert mjconvert/tests/resources/mjlog mjconvert/tests/resources/mjxproto --to-mjxproto --verbose && cat mjconvert/tests/resources/mjxproto/* | wc -l
	rm mjconvert/tests/resources/mjxproto/*.json
	mjconvert mjconvert/tests/resources/mjlog mjconvert/tests/resources/mjxproto --to-mjxproto --compress --verbose && cat mjconvert/tests/resources/mjxproto/* | wc -l
	rm mjconvert/tests/resources/mjxproto/*.json
	mjconvert mjconvert/tests/resources/mjlog mjconvert/tests/resources/mjxproto --to-mjxproto-raw --verbose && cat mjconvert/tests/resources/mjxproto/* | wc -l
	echo "From mjxproto => mjlog_recovered"
	cat mjconvert/tests/resources/mjxproto/*.json | mjconvert --to-mjlog --verbose | wc -l
	mjconvert mjconvert/tests/resources/mjxproto mjconvert/tests/resources/mjlog_recovered --to-mjlog --verbose && cat mjconvert/tests/resources/mjlog_recovered/* | wc -l
	echo "Check diff"
	python3 mjconvert/diff.py mjconvert/tests/resources/mjlog mjconvert/tests/resources/mjlog_recovered
	echo "Clean up"
	rm -rf mjconvert/tests/resources/mjxproto
	rm -rf mjconvert/tests/resources/mjlog_recovered
	git checkout -- mjconvert/tests/resources
