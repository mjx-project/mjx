.PHONY: proto requirements install uninstall clean pytest test-cli

clean:
	rm -rf ./mjxproto/*pb2*
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

formats:
	black mjx
	blackdoc mjx
	isort mjx

checks:
	black --check --diff mjx
	blackdoc --check mjx
	isort --check --diff mjx
	flake8 --config pyproject.toml --ignore E203,E501,W503 mjx
	# mypy --config pyproject.toml mjx

proto: ../mjx/include/mjx/internal/mjx.proto
	python3 -m grpc_tools.protoc -I ../mjx/internal --python_out=./mjxproto/ --grpc_python_out=./mjxproto/ --mypy_out=./mjxproto/ mjx.proto

requirements:
	python3 -m pip install -r requirements.txt

uninstall:
	python3 -m pip uninstall mjx -y

install: requirements proto
	python3 setup.py install

pytest: proto
	python3 -m pytest --doctest-modules

test-cli: uninstall clean install
	echo "From mjlog => mjxproto"
	cat mjx/tests/resources/mjlog/*.mjlog | mjx convert --to-mjxproto --verbose | wc -l
	cat mjx/tests/resources/mjlog/*.mjlog | mjx convert --to-mjxproto --compress --verbose | wc -l
	cat mjx/tests/resources/mjlog/*.mjlog | mjx convert --to-mjxproto-raw --verbose | wc -l
	mjx convert mjx/tests/resources/mjlog mjx/tests/resources/mjxproto --to-mjxproto --verbose && cat mjx/tests/resources/mjxproto/* | wc -l
	rm mjx/tests/resources/mjxproto/*.json
	mjx convert mjx/tests/resources/mjlog mjx/tests/resources/mjxproto --to-mjxproto --compress --verbose && cat mjx/tests/resources/mjxproto/* | wc -l
	rm mjx/tests/resources/mjxproto/*.json
	mjx convert mjx/tests/resources/mjlog mjx/tests/resources/mjxproto --to-mjxproto-raw --verbose && cat mjx/tests/resources/mjxproto/* | wc -l
	echo "From mjxproto => mjlog_recovered"
	cat mjx/tests/resources/mjxproto/*.json | mjx convert --to-mjlog --verbose | wc -l
	mjx convert mjx/tests/resources/mjxproto mjx/tests/resources/mjlog_recovered --to-mjlog --verbose && cat mjx/tests/resources/mjlog_recovered/* | wc -l
	echo "Check diff"
	python3 mjx/diff.py mjx/tests/resources/mjlog mjx/tests/resources/mjlog_recovered
	echo "Clean up"
	rm -rf mjx/tests/resources/mjxproto
	rm -rf mjx/tests/resources/mjlog_recovered
	git checkout -- mjx/tests/resources
