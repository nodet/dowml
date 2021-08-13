test_locally_built:
	python -m pip install  --force-reinstall ../../dist/dowml-0.1.0-py3-none-any.whl
	pip install ibm-watson-machine-learning
	python list_jobs.py
	dowml -c 'jobs'

test_pypi_test_build:
	python -m pip install  --force-reinstall --extra-index-url -i https://test.pypi.org/simple/ dowml
	python list_jobs.py
	dowml -c 'jobs'
