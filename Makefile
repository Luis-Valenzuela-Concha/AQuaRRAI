.PHONY: clear-base-venv clear-pyralysis-venv clear-venv setup-base-venv install-base setup-pyralysis-venv install-pyralysis install-all

# Clear the environment
clear-base-venv:
	rm -rf venv_base

clear-pyralysis-venv:
	rm -rf venv_pyralysis

clear-venv: clear-base-venv clear-pyralysis-venv

# Set up the virtual environment
setup-base-venv:
	mkdir -p envs
	test -d venv_base || python -m venv venv_base
	venv_base/bin/pip install --upgrade pip

install-base: setup-base-venv
	venv_base/bin/pip install -r requirements/base.txt

setup-pyralysis-venv: setup-base-venv
	test -d venv_pyralysis || python -m venv venv_pyralysis
	venv_pyralysis/bin/pip install --upgrade pip

install-pyralysis: setup-pyralysis-venv install-base
	venv_pyralysis/bin/pip install -r requirements/pyralysis.txt

install-all: install-base install-pyralysis