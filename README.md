# dowml
A library and command line client to use Decision Optimization on WML

The class `DOWMLClient` provides an API to upload Decision Optimization models (CPLEX, CP Optimizer, OPL or docplex) to WML, check their status, and download results.  The script `cli.py` is an interactive program on top of that library.

In order to use either of them, you need to provide IBM Cloud credentials.
1. By default, `DOWMLClient` (and therefore the Interactive) look for these credentials in an environment variable named `WML_CREDENTIALS`. This variable shoud have a value looking like `{'apikey': '<apikey>', 'url': 'https://us-south.ml.cloud.ibm.com'}`.
2. As an alternative, you can specify a file name as argument to `DOWMLClient.__init__`. The credentials will then be read from that file instead of the environment variable. Accordingly, the Interactive has a command line option `--wml-cred-file` that must be followed by the path of the file.

Here's a sample session:
```
$ python3 cli.py --wml-cred-file xavier-wml-cred.txt
2021-02-14 14:22:29,753 Looking for credentials in file 'xavier-wml-cred.txt'...
2021-02-14 14:22:29,753 Found credential string.
2021-02-14 14:22:29,753 Credentials have the expected structure.

Decision Optimization in WML Interactive.
Submit and manage Decision Optimization models interactively.

Type ? for a list of commands.

Most commands need an argument that can be either a job id, or the number
of the job, as displayed by the 'jobs' command.  If a command requires a
job id, but none is specified, the last one is used.

dowml> help

Documented commands (type help <topic>):
========================================
cancel  delete  details  exit  help  jobs  log  size  solve  type  wait

dowml>
```
