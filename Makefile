quick_tests = tests/test_lib.py tests/test_cli.py tests/test_bugs.py
slow_tests = tests/test_full.py
all_tests = $(quick_tests) $(slow_tests)

quick:
	python -m unittest $(quick_tests)
	rm -rf a_details.json

# A very basic test to confirm the client version number and that
# credentials work
basic:
	python3 dowml.py -vv -c jobs

# Actual tests with checks
slow:
	python -m unittest $(slow_tests)

cover:
	coverage run -m unittest $(quick_tests)
	rm -rf a_details.json
	coverage html
	open htmlcov/dowmllib_py.html

# This really exercises as much as possible
# but doesn't check the results
fulltests:
	python3 dowml.py -c help type size 'inline yes' \
       'solve examples/afiro.mps' jobs wait log delete \
       'type docplex' 'solve examples/markshare.py examples/markshare1.mps.gz' wait jobs output 'details full' delete \
       'inline no' 'type cplex' 'solve examples/afiro.mps' jobs wait details delete \
       'type docplex' 'solve examples/markshare.py examples/markshare1.mps.gz' 'details names' wait delete

delete-space:
	PYTHONPATH=. python3 tests/delete_space.py