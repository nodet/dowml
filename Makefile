.ONESHELL:
.PHONY: tests

tests:
	$(MAKE) -C tests quick basic slow fulltests

lint:
	flake8 src tests/*.py

build: lint
	python3 -m build

install: build
	python -m pip uninstall --yes dowml
	python -m pip install  dist/dowml-*-py3-none-any.whl

upload-on-test:
	util/upload-on-test.sh

upload-on-real:
	util/upload-on-test.sh real
