rebuild_venv:
	rm -rf test-pypi-venv/
	python3 -m venv test-pypi-venv
	. test-pypi-venv/bin/activate    ;\
	pip install --upgrade pip

test_locally_built:
	. test-pypi-venv/bin/activate                                                     ;\
	python -m pip install  ../../dist/dowml-*.whl                                     ;\
	python list_jobs.py                                                               ;\
	dowml -c 'jobs'

test_pypi_test_build:
	. test-pypi-venv/bin/activate                                                                          ;\
	python -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple dowml ;\
	python list_jobs.py                                                                                    ;\
	dowml -c 'jobs'

test_pypi_build:
	. test-pypi-venv/bin/activate  ;\
	python -m pip install dowml    ;\
	python list_jobs.py            ;\
	dowml -c 'jobs'
