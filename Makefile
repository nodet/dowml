.ONESHELL:
.PHONY: tests

tests:
	$(MAKE) -C tests quick basic slow fulltests

lint:
	flake8 src tests/*.py

build: lint
	python3 -m build

upload-on-test:
	util/upload-on-test.sh

upload-on-real:
	util/upload-on-test.sh real
