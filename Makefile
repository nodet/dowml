all:
	$(MAKE) -C tests quick basic slow fulltests

build:
	python3 -m build