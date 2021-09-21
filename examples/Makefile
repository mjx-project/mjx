.PHONY: checks formats install test

venv:
	python3 -m venv venv

formats:
	black .
	blackdoc .
	isort .

checks:
	black --check --diff .
	blackdoc --check .
	isort --check --diff .

install:
	python3 -m pip install --upgrade pip
	python3 -m pip install mjx gym pettingzoo "ray[rllib]" torch
	python3 -m pip install black blackdoc isort

test:
	python3 rllib_example.py
	python3 pettingzoo_example.py
