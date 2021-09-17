.PHONY: proto requirements clean pytest test-cli

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
	python3 -m grpc_tools.protoc -I ../mjx/include/mjx/internal --python_out=./mjxproto/ --grpc_python_out=./mjxproto/ --mypy_out=./mjxproto/ mjx.proto

requirements:
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt

pytest: requirements proto
	python3 -m pytest --doctest-modules

test-cli:
	echo "From mjlog => mjxproto"
	cat tests/resources/mjlog/*.mjlog | mjx convert --to-mjxproto --verbose | wc -l
	cat tests/resources/mjlog/*.mjlog | mjx convert --to-mjxproto --compress --verbose | wc -l
	cat tests/resources/mjlog/*.mjlog | mjx convert --to-mjxproto-raw --verbose | wc -l
	mjx convert tests/resources/mjlog tests/resources/mjxproto --to-mjxproto --verbose && cat tests/resources/mjxproto/* | wc -l
	rm tests/resources/mjxproto/*.json
	mjx convert tests/resources/mjlog tests/resources/mjxproto --to-mjxproto --compress --verbose && cat tests/resources/mjxproto/* | wc -l
	rm tests/resources/mjxproto/*.json
	mjx convert tests/resources/mjlog tests/resources/mjxproto --to-mjxproto-raw --verbose && cat tests/resources/mjxproto/* | wc -l
	echo "From mjxproto => mjlog_recovered"
	cat tests/resources/mjxproto/*.json | mjx convert --to-mjlog --verbose | wc -l
	mjx convert tests/resources/mjxproto tests/resources/mjlog_recovered --to-mjlog --verbose && cat tests/resources/mjlog_recovered/* | wc -l
	echo "Check diff"
	python3 mjx/diff.py tests/resources/mjlog tests/resources/mjlog_recovered
	echo "Clean up"
	rm -rf tests/resources/mjxproto
	rm -rf tests/resources/mjlog_recovered
	git checkout -- tests/resources
