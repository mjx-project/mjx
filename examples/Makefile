.PHONY: install test

venv:
	python3 -m venv venv

install:
	python3 -m pip install --upgrade pip
	python3 -m pip install mjx gym pettingzoo "ray[rllib]" torch

test:
	python3 rllib_example.py
	python3 pettingzoo_example.py
