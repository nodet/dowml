tests:
	$(MAKE) -C tests quick basic slow fulltests

build:
	python3 -m build

upload-on-test:
	python3 -m twine upload --repository testpypi dist/*