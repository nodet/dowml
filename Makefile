quick:
	python -m unittest tests/test_lib.py tests/test_cli.py
	rm -rf a_details.json

cover:
	coverage run -m unittest tests/test_lib.py tests/test_cli.py
	rm -rf a_details.json
	coverage html
	open htmlcov/dowmllib_py.html

#
# A very basic test to confirm the client version number and that
# credentials work
#
basic:
	python3 dowml.py -vv -c jobs

#
# This really tests as much as possible
#
fulltests:
	python3 dowml.py -c help type size 'inline yes' \
       'solve examples/afiro.mps' jobs wait log delete \
       'type docplex' 'solve examples/markshare.py examples/markshare1.mps.gz' wait jobs output 'details full' delete \
       'inline no' 'type cplex' 'solve examples/afiro.mps' jobs wait details delete \
       'type docplex' 'solve examples/markshare.py examples/markshare1.mps.gz' 'details names' wait delete