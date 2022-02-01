rebuild_venv:
	rm -rf test-pypi-venv/
	python3 -m venv test-pypi-venv
	. test-pypi-venv/bin/activate    ;\
	pip install --upgrade pip

test_locally_built:
	. test-pypi-venv/bin/activate                                                     ;\
	python -m pip install  ../../dist/dowml-*.whl                                     ;\
    pip show dowml                                                                    ;\
	python list_jobs.py                                                               ;\
	dowml -c 'jobs'

test_pypi_test_build:
	. test-pypi-venv/bin/activate                                                                          ;\
	python -m pip install ibm-watson-machine-learning                                                      ;\
	python -m pip install -i https://pypi.org/simple --extra-index-url https://test.pypi.org/simple/ dowml ;\
    pip show dowml                                                                                         ;\
	python list_jobs.py                                                                                    ;\
	dowml -c 'jobs'

test_pypi_build:
	. test-pypi-venv/bin/activate  ;\
	python -m pip install dowml    ;\
    pip show dowml                 ;\
	python list_jobs.py            ;\
	dowml -c 'jobs'
