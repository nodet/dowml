tests:
	PYTHONPATH=$PYTHONPATH:.. python -m unittest test_lib.py test_cli.py
	rm -rf a_details.json

cover:
	PYTHONPATH=$PYTHONPATH:.. coverage run -m unittest test_lib.py test_cli.py
	rm -rf a_details.json
	coverage html
	open htmlcov/_Users_nodet_Documents_work_git_master_dowml_dowmllib_py.html

fulltests:
	python3 ../dowml.py -c help type size 'inline yes' \
       'solve ../examples/afiro.mps' jobs wait log delete \
       'type docplex' 'solve ../examples/markshare.py ../examples/markshare1.mps.gz' wait jobs output 'details full' delete \
       'inline no' 'type cplex' 'solve ../examples/afiro.mps' jobs wait details delete \
       'type docplex' 'solve ../examples/markshare.py ../examples/markshare1.mps.gz' 'details names' wait delete