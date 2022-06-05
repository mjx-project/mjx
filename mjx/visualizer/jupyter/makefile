venv:
	python3.9 -m venv venv
	source venv/bin/activate
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt

run:
	jupyter notebook

format:
	black .
	isort .

.PHONY: venv run format
