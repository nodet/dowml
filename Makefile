.ONESHELL:
.PHONY: tests

tests:
	$(MAKE) -C tests quick basic slow fulltests

SRC=src/dowml/*.py

lint: $(SRC)
	flake8 $(SRC) tests/*.py

DISTRIB=dist/dowml-*-py3-none-any.whl

$(DISTRIB): $(SRC)
	rm -rf dist/*
	python3 -m build

build: lint $(DISTRIB)
	touch build

install: build
	python -m pip uninstall --yes dowml
	python -m pip install  dist/dowml-*-py3-none-any.whl
	touch install

upload-on-test:
	util/upload-on-test.sh

upload-on-real:
	util/upload-on-test.sh real
