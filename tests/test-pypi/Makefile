rebuild_venv:
	rm -rf test-pypi-venv/
	python3 -m venv test-pypi-venv
	. test-pypi-venv/bin/activate    ;\
	pip install --upgrade pip

test_locally_built:
	. test-pypi-venv/bin/activate                                                     ;\
	python -m pip install  --force-reinstall ../../dist/dowml-*.whl  ;\
	python list_jobs.py                                                               ;\
	dowml -c 'jobs'

test_pypi_test_build:
	source test-pypi-venv/bin/activate
	python -m pip install  --force-reinstall --extra-index-url -i https://test.pypi.org/simple/ dowml
	python list_jobs.py
	dowml -c 'jobs'
