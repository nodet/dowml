# dowml
A library and command line client to use Decision Optimization on WML

The class `DOWMLLib` provides an API to upload Decision Optimization models (CPLEX, CP Optimizer, OPL or docplex) to WML, check their status, and download results.  The script `dowml.py` is an interactive program on top of that library.

In order to use either of them, you need to provide IBM Cloud credentials.
1. By default, `DOWMLLib` (and therefore the Interactive) look for these credentials in an environment variable named `WML_CREDENTIALS`. This variable shoud have a value looking like `{'apikey': '<apikey>', 'url': 'https://us-south.ml.cloud.ibm.com'}`.
2. As an alternative, you can specify a file name as argument to `DOWMLLib.__init__`. The credentials will then be read from that file instead of the environment variable. Accordingly, the Interactive has a command line option `-w` (or `--wml-cred-file`) that must be followed by the path of the file.

Here's a sample session:
```
$ python3 dowml.py -h
usage: dowml.py [-h] [-w WML_CRED_FILE]

Interactive program for DO on WML

optional arguments:
  -h, --help            show this help message and exit
  -w WML_CRED_FILE, --wml-cred-file WML_CRED_FILE
                        Name of the file from which to read WML credentials. If not specified, credentials are read
                        from environment variable $WML_CREDENTIALS.
$
$
$ python3 dowml.py -w xavier-wml-cred.txt
2021-02-16 13:51:14,377 Looking for credentials in file 'xavier-wml-cred.txt'...
2021-02-16 13:51:14,378 Found credential string.
2021-02-16 13:51:14,379 Credentials have the expected structure.

Decision Optimization in WML Interactive.
Submit and manage Decision Optimization models interactively.

Type ? for a list of commands.

Most commands need an argument that can be either a job id, or the number
of the job, as displayed by the 'jobs' command.  If a command requires a
job id, but none is specified, the last one is used.

dowml> help

Documented commands (type help <topic>):
========================================
cancel  delete  details  exit  help  jobs  log  output  size  solve  type  wait

dowml> type
Current model type: cplex. Known types: cplex, cpo, opl, docplex
dowml> size
Current size: S. Known sizes: S, M, XL
dowml> solve examples/afiro.mps
2021-02-16 13:53:51,057 Creating the connexion...
2021-02-16 13:53:54,439 Creating the connexion succeeded.  Client version is 1.0.45
2021-02-16 13:53:54,439 Creating the connexion...
2021-02-16 13:53:57,127 Creating the connexion succeeded.  Client version is 1.0.45
2021-02-16 13:53:57,127 Fetching existing spaces...
2021-02-16 13:53:58,628 Got the list. Looking for space named 'DOWMLClient-space'
2021-02-16 13:53:58,628 Found it.
2021-02-16 13:53:58,629 Space id: 15f1a4b1-1e2b-4a60-8b4f-cf540ca65d36
2021-02-16 13:53:58,629 Setting default space...
2021-02-16 13:54:04,915 Done.
2021-02-16 13:54:04,915 Fetching existing spaces...
2021-02-16 13:54:06,039 Got the list. Looking for space named 'DOWMLClient-space'
2021-02-16 13:54:06,039 Found it.
2021-02-16 13:54:06,039 Space id: 15f1a4b1-1e2b-4a60-8b4f-cf540ca65d36
2021-02-16 13:54:06,039 Setting default space...
2021-02-16 13:54:10,894 Done.
2021-02-16 13:54:10,894 Getting deployments...
2021-02-16 13:54:12,294 Done.
2021-02-16 13:54:12,294 Got the list. Looking for deployment named 'DOWMLClient-deployment-cplex-S'
2021-02-16 13:54:12,294 Found it.
2021-02-16 13:54:12,294 Deployment id: 82d98524-d9de-4bda-bb82-224305ab2c5a
2021-02-16 13:54:12,295 Creating the job...
2021-02-16 13:54:16,563 Done. Getting its id...
2021-02-16 13:54:16,563 Job id: 3746c20a-cbfa-4922-9df7-29652d8f1b89
Job id: 3746c20a-cbfa-4922-9df7-29652d8f1b89
dowml> jobs
     #   status     id                                    creation date             inputs
=>   1:  completed  3746c20a-cbfa-4922-9df7-29652d8f1b89  2021-02-16T12:54:16.051Z  afiro.mps
dowml> log
2021-02-16 13:54:54,002 Fetching output...
2021-02-16 13:54:55,305 Done.
[2021-02-16T12:54:16Z, INFO] CPLEX version 12100000
[2021-02-16T12:54:16Z, WARNING] Changed parameter CPX_PARAM_THREADS from 0 to 1
[2021-02-16T12:54:16Z, INFO] Param[1,067] = 1
[2021-02-16T12:54:16Z, INFO] Param[1,130] = UTF-8
[2021-02-16T12:54:16Z, INFO] Param[1,132] = -1
[2021-02-16T12:54:16Z, INFO]
[2021-02-16T12:54:16Z, INFO] Selected objective sense:  MINIMIZE
[2021-02-16T12:54:16Z, INFO] Selected objective  name:  obj
[2021-02-16T12:54:16Z, INFO] Selected RHS        name:  rhs
[2021-02-16T12:54:16Z, INFO] Version identifier: 12.10.0.0 | 2020-01-09 | 0d94640
[2021-02-16T12:54:16Z, INFO] CPXPARAM_Threads                                 1
[2021-02-16T12:54:16Z, INFO] CPXPARAM_Output_CloneLog                         -1
[2021-02-16T12:54:16Z, INFO] CPXPARAM_Read_APIEncoding                        "UTF-8"
[2021-02-16T12:54:16Z, INFO] Tried aggregator 1 time.
[2021-02-16T12:54:16Z, INFO] LP Presolve eliminated 9 rows and 10 columns.
[2021-02-16T12:54:16Z, INFO] Aggregator did 7 substitutions.
[2021-02-16T12:54:16Z, INFO] Reduced LP has 11 rows, 15 columns, and 37 nonzeros.
[2021-02-16T12:54:16Z, INFO] Presolve time = 0.00 sec. (0.03 ticks)
[2021-02-16T12:54:16Z, INFO]
[2021-02-16T12:54:16Z, INFO] Iteration log . . .
[2021-02-16T12:54:16Z, INFO] Iteration:     1   Scaled dual infeas =             1.200000
[2021-02-16T12:54:16Z, INFO] Iteration:     5   Dual objective     =          -464.753143
[2021-02-16T12:54:17Z, INFO] There are no bound infeasibilities.
[2021-02-16T12:54:17Z, INFO] There are no reduced-cost infeasibilities.
[2021-02-16T12:54:17Z, INFO] Max. unscaled (scaled) Ax-b resid.          = 1.77636e-14 (1.77636e-14)
[2021-02-16T12:54:17Z, INFO] Max. unscaled (scaled) c-B'pi resid.        = 5.55112e-17 (5.55112e-17)
[2021-02-16T12:54:17Z, INFO] Max. unscaled (scaled) |x|                  = 500 (500)
[2021-02-16T12:54:17Z, INFO] Max. unscaled (scaled) |slack|              = 500 (500)
[2021-02-16T12:54:17Z, INFO] Max. unscaled (scaled) |pi|                 = 0.942857 (1.88571)
[2021-02-16T12:54:17Z, INFO] Max. unscaled (scaled) |red-cost|           = 10 (10)
[2021-02-16T12:54:17Z, INFO] Condition number of scaled basis            = 1.5e+01
[2021-02-16T12:54:17Z, INFO] optimal (1)
dowml> type docplex
dowml> solve examples/markshare.py examples/markshare1.mps.gz
2021-02-16 13:57:04,796 Getting deployments...
2021-02-16 13:57:05,936 Done.
2021-02-16 13:57:05,936 Got the list. Looking for deployment named 'DOWMLClient-deployment-docplex-S'
2021-02-16 13:57:05,936 Found it.
2021-02-16 13:57:05,936 Deployment id: 71653010-db88-4ca9-a8a9-857ee66a93e0
2021-02-16 13:57:05,938 Creating the job...
2021-02-16 13:57:10,058 Done. Getting its id...
2021-02-16 13:57:10,058 Job id: adb0e1f9-d765-45e1-9dc3-b3ad3088fd2f
Job id: adb0e1f9-d765-45e1-9dc3-b3ad3088fd2f
dowml> jobs
     #   status     id                                    creation date             inputs
     1:  completed  3746c20a-cbfa-4922-9df7-29652d8f1b89  2021-02-16T12:54:16.051Z  afiro.mps
=>   2:  completed  adb0e1f9-d765-45e1-9dc3-b3ad3088fd2f  2021-02-16T12:57:09.423Z  markshare.py, markshare1.mps.gz
dowml> output
2021-02-16 13:58:20,530 Fetching output...
2021-02-16 13:58:21,762 Done.
Storing adb0e1f9-d765-45e1-9dc3-b3ad3088fd2f_solution.json
Storing adb0e1f9-d765-45e1-9dc3-b3ad3088fd2f_log.txt
```
