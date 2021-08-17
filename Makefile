.ONESHELL:

tests:
	$(MAKE) -C tests quick basic slow fulltests

build:
	python3 -m build

upload-on-test:
	util/upload-on-test.sh

upload-on-real:
	util/upload-on-test.sh real
