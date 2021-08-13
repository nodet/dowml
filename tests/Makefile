export PYTHONPATH=$PYTHONPATH:../src/dowml

quick_tests = test_lib.py test_cli.py test_bugs.py
slow_tests = test_full.py
all_tests = $(quick_tests) $(slow_tests)

quick:
	python -m unittest $(quick_tests)
	rm -rf a_details.json

# A very basic test to confirm the client version number and that
# credentials work
basic:
	python3 ../src/dowml/dowml.py -vv -c jobs

# Actual tests with checks
slow:
	python -m unittest $(slow_tests)

cover:
	coverage run -m unittest $(all_tests)
	rm -rf a_details.json
	coverage html
	open htmlcov/_Users_nodet_Documents_work_git_master_dowml_src_dowml_dowmllib_py.html

# This really exercises as much as possible
# but doesn't check the results
fulltests:
	python3 ../src/dowml/dowml.py -c help type size 'inline yes' \
       'solve ../examples/afiro.mps' jobs wait log delete \
       'type docplex' 'solve ../examples/markshare.py ../examples/markshare1.mps.gz' wait jobs output 'details full' delete \
       'inline no' 'type cplex' 'solve ../examples/afiro.mps' jobs wait details delete \
       'type docplex' 'solve ../examples/markshare.py ../examples/markshare1.mps.gz' 'details names' wait delete

delete-space:
	PYTHONPATH=. python3 delete_space.py