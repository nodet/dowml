# dowml
A library and command line client to use Decision Optimization on WML

*Note that this tool is not an official IBM product, and is not supported by IBM. This is educational material, 
use at your own risks.*

## tldr;

```
$ pip install dowml
$ cat my_credentials.txt
{
    'apikey': '<apikey>',
    'region': 'us-south',
    'cos_resource_crn': 'crn:v1:bluemix:public:cloud-object-storage:global:a/76260f9...',
    'ml_instance_crn': 'crn:v1:bluemix:public:pm-20:eu-de:a/76260f...'
}
$ dowml -w my-credentials.txt
dowml> solve examples/afiro.mps
dowml> wait
dowml> log
dowml> exit
```

## Introduction

The class `DOWMLLib` provides an API to upload Decision Optimization models (CPLEX, CP Optimizer, OPL or docplex) to WML, check their status, and download results.  The script `dowml.py` is an interactive program on top of that library.

In order to use either of them, you need to provide IBM Cloud credentials.
- By default, `DOWMLLib` (and therefore the Interactive) look for these credentials in an environment variable named `DOWML_CREDENTIALS`. This variable shoud have a value looking like
```
{
    'apikey': '<apikey>',
    'region': 'us-south',
    'cos_resource_crn': 'crn:v1:bluemix:public:cloud-object-storage:global:a/76260f9...',
    'ml_instance_crn': 'crn:v1:bluemix:public:pm-20:eu-de:a/76260f...',
}
```
See below for how/where to get these credentials.
- As an alternative, you can specify a file name as argument to 
`DOWMLLib.__init__`. The credentials will then be read from that file instead 
of the environment variable. Accordingly, the Interactive has a command line 
option `-w` (or `--wml-cred-file`) that must be followed by the path of the file.
- Finally, if none of the above options are used, the code will look for
environment variable `DOWML_CREDENTIALS_FILE`. If it exists, it must be the path
to a file that contains credentials such as the ones above.

Here's a sample session:
```
$ dowml -h
usage: interactive.py [-h] [--wml-cred-file WML_CRED_FILE] [--verbose]
                      [--commands [COMMANDS [COMMANDS ...]]] [--input] [--space SPACE] [--url URL]                      [--api-key API_KEY] [--region REGION]Decision Optimization in WML Interactive, version 1.9.0.Submit and manage Decision Optimization models interactively.(c) Copyright Xavier Nodet, 2022

optional arguments:
  -h, --help            show this help message and exit
  --wml-cred-file WML_CRED_FILE, -w WML_CRED_FILE
                        Name of the file from which to read WML credentials. If not specified,
                        credentials are read from environment variable $DOWML_CREDENTIALS. If no such
                        variable exists, but variable $DOWML_CREDENTIALS_FILE exists, tries to read
                        that file.
  --verbose, -v         Verbose mode. Causes the program to print debugging messages about its
                        progress. Multiple -v options increase the verbosity. The maximum is 4.
  --commands [COMMANDS [COMMANDS ...]], -c [COMMANDS [COMMANDS ...]]
                        Carries out the specified commands. Each command is executed as if it had been
                        specified at the prompt. The program stops after last command, unless --input
                        is used.
  --input, -i           Prompts for new input commands even if some commands have been specified as
                        arguments using --commands.
  --space SPACE, -s SPACE
                        Id of the space to connect to. Takes precedence over the one specified in the
                        credentials under the 'space_id' key, if any.
  --url URL, -u URL     URL to use for the Machine Learning service. Takes precedence over the one
                        specified in the credentials under the 'url' key, if any. Incompatible with
                        --region argument.
  --api-key API_KEY, -k API_KEY
                        API key to use to connect to WML. Takes precedence over the one specified in
                        the credentials under the 'apikey' key, if any.
  --region REGION, -r REGION
                        Region to use for the Machine Learning service. Takes precedence over the
                        region or URL specified in the credentials, if any. Incompatible with --url
                        argument. Possible values for the region are ['us-south', 'eu-de', 'eu-gb',
                        'jp-tok'].
$
$
$ dowml -c help type size 'inputs inline' 'solve examples/afiro.mps' jobs wait jobs log 'type docplex' 'solve examples/markshare.py examples/markshare1.mps.gz' wait jobs dump 'shell ls -l *-*-*-*-*' 'delete *'

Decision Optimization in WML Interactive, version 1.9.0.
Submit and manage Decision Optimization models interactively.
(c) Copyright Xavier Nodet, 2022

Type ? for a list of commands.

Most commands need an argument that can be either a job id, or the number
of the job, as displayed by the 'jobs' command.  If a command requires a
job id, but none is specified, the last one is used.

dowml> help

Documented commands (type help <topic>):
========================================
cancel  details  exit  inline  jobs  output   shell  solve   time  version
delete  dump     help  inputs  log   outputs  size   status  type  wait

dowml> type
Current model type: cplex.
Known types: cplex, cpo, opl, docplex.
dowml> size
Current size: S.
Known sizes: S, M, L, XL.
dowml> inputs inline
dowml> solve examples/afiro.mps
Job id: cd494377-4843-40a4-ae84-ede7f8c16eda
dowml> jobs
     #  status      id                                    creation date        type     ver.   size  inputs
=>   1: queued      cd494377-4843-40a4-ae84-ede7f8c16eda  2022-06-28 11:59:06  cplex    22.1   S     afiro.mps
dowml> wait
Job is running.
Job is completed.
Job has finished with status 'completed'.
dowml> jobs
     #  status      id                                    creation date        type     ver.   size  inputs
=>   1: completed   cd494377-4843-40a4-ae84-ede7f8c16eda  2022-06-28 11:59:06  cplex    22.1   S     afiro.mps
dowml> log
[2022-06-28T09:59:08Z, INFO] CPLEX version 22010000
[2022-06-28T09:59:08Z, WARNING] Changed parameter CPX_PARAM_THREADS from 0 to 1
[2022-06-28T09:59:08Z, INFO] Param[1,067] = 1
[2022-06-28T09:59:08Z, INFO] Param[1,130] = UTF-8
[2022-06-28T09:59:08Z, INFO] Param[1,132] = -1
[2022-06-28T09:59:08Z, INFO]
[2022-06-28T09:59:08Z, INFO] Selected objective sense:  MINIMIZE
[2022-06-28T09:59:08Z, INFO] Selected objective  name:  obj
[2022-06-28T09:59:08Z, INFO] Selected RHS        name:  rhs
[2022-06-28T09:59:08Z, INFO] Version identifier: 22.1.0.0 | 2022-03-30 | 54982fbec
[2022-06-28T09:59:08Z, INFO] CPXPARAM_Threads                                 1
[2022-06-28T09:59:08Z, INFO] CPXPARAM_Output_CloneLog                         -1
[2022-06-28T09:59:08Z, INFO] CPXPARAM_Read_APIEncoding                        "UTF-8"
[2022-06-28T09:59:08Z, INFO] Tried aggregator 1 time.
[2022-06-28T09:59:08Z, INFO] LP Presolve eliminated 9 rows and 10 columns.
[2022-06-28T09:59:08Z, INFO] Aggregator did 7 substitutions.
[2022-06-28T09:59:08Z, INFO] Reduced LP has 11 rows, 15 columns, and 37 nonzeros.
[2022-06-28T09:59:08Z, INFO] Presolve time = 0.00 sec. (0.03 ticks)
[2022-06-28T09:59:08Z, INFO]
[2022-06-28T09:59:08Z, INFO] Iteration log . . .
[2022-06-28T09:59:08Z, INFO] Iteration:     1   Scaled dual infeas =             1.200000
[2022-06-28T09:59:08Z, INFO] Iteration:     5   Dual objective     =          -464.753143
[2022-06-28T09:59:09Z, INFO] There are no bound infeasibilities.
[2022-06-28T09:59:09Z, INFO] There are no reduced-cost infeasibilities.
[2022-06-28T09:59:09Z, INFO] Max. unscaled (scaled) Ax-b resid.          = 1.77636e-14 (1.77636e-14)
[2022-06-28T09:59:09Z, INFO] Max. unscaled (scaled) c-B'pi resid.        = 5.55112e-17 (5.55112e-17)
[2022-06-28T09:59:09Z, INFO] Max. unscaled (scaled) |x|                  = 500 (500)
[2022-06-28T09:59:09Z, INFO] Max. unscaled (scaled) |slack|              = 500 (500)
[2022-06-28T09:59:09Z, INFO] Max. unscaled (scaled) |pi|                 = 0.942857 (1.88571)
[2022-06-28T09:59:09Z, INFO] Max. unscaled (scaled) |red-cost|           = 10 (10)
[2022-06-28T09:59:09Z, INFO] Condition number of scaled basis            = 1.5e+01
[2022-06-28T09:59:09Z, INFO] optimal (1)
dowml> type docplex
dowml> solve examples/markshare.py examples/markshare1.mps.gz
Job id: 6520e72b-727c-4bfe-adb5-a40d96cf5910
dowml> wait
Job is queued.
Job is running..
[2022-06-28T09:59:16Z, WARNING] Python 3.9 is used as default with pandas 1.3 libraries.
[2022-06-28T09:59:17Z, INFO] Reading markshare1.mps.gz...
.
[2022-06-28T09:59:16Z, WARNING] Python 3.9 is used as default with pandas 1.3 libraries.
[2022-06-28T09:59:17Z, INFO] Reading markshare1.mps.gz...
.
[2022-06-28T09:59:16Z, WARNING] Python 3.9 is used as default with pandas 1.3 libraries.
[2022-06-28T09:59:17Z, INFO] Reading markshare1.mps.gz...
.
[2022-06-28T09:59:17Z, INFO]
[2022-06-28T09:59:17Z, INFO]         Nodes                                         Cuts/
[2022-06-28T09:59:17Z, INFO]    Node  Left     Objective  IInf  Best Integer    Best Bound    ItCnt     Gap
[2022-06-28T09:59:17Z, INFO]
[2022-06-28T09:59:17Z, INFO] *     0+    0                         7286.0000        0.0000           100.00%
[2022-06-28T09:59:17Z, INFO]       0     0        0.0000     6     7286.0000        0.0000       11  100.00%
[2022-06-28T09:59:17Z, INFO] *     0+    0                          263.0000        0.0000           100.00%
[2022-06-28T09:59:17Z, INFO] *     0+    0                          230.0000        0.0000           100.00%
[2022-06-28T09:59:17Z, INFO]       0     0        0.0000     7      230.0000      Cuts: 15       15  100.00%
[2022-06-28T09:59:17Z, INFO]       0     0        0.0000     7      230.0000      Cuts: 16       23  100.00%
[2022-06-28T09:59:17Z, INFO] *     0+    0                          193.0000        0.0000           100.00%
[2022-06-28T09:59:17Z, INFO] Detecting symmetries...
[2022-06-28T09:59:17Z, INFO]       0     2        0.0000     7      193.0000        0.0000       23  100.00%
[2022-06-28T09:59:17Z, INFO] Elapsed time = 0.01 sec. (2.91 ticks, tree = 0.01 MB, solutions = 4)
[2022-06-28T09:59:17Z, INFO] *    70+   59                          166.0000        0.0000           100.00%
[2022-06-28T09:59:17Z, INFO] *    80+   67                          132.0000        0.0000           100.00%
[2022-06-28T09:59:17Z, INFO] *   190+  155                          111.0000        0.0000           100.00%
[2022-06-28T09:59:17Z, INFO] *   220+  166                           96.0000        0.0000           100.00%
[2022-06-28T09:59:17Z, INFO] *   320+  240                           71.0000        0.0000           100.00%
[2022-06-28T09:59:17Z, INFO] *   420+  305                           67.0000        0.0000           100.00%
[2022-06-28T09:59:17Z, INFO] *   420+  303                           66.0000        0.0000           100.00%
[2022-06-28T09:59:17Z, INFO] *   491   310      integral     0       38.0000        0.0000     1112  100.00%
[2022-06-28T09:59:17Z, INFO]
[2022-06-28T09:59:17Z, INFO] Performing restart 1
[2022-06-28T09:59:17Z, INFO]
[2022-06-28T09:59:17Z, INFO] Repeating presolve.
[2022-06-28T09:59:17Z, INFO] Tried aggregator 1 time.
[2022-06-28T09:59:17Z, INFO] Reduced MIP has 6 rows, 56 columns, and 306 nonzeros.
[2022-06-28T09:59:17Z, INFO] Reduced MIP has 50 binaries, 6 generals, 0 SOSs, and 0 indicators.
[2022-06-28T09:59:17Z, INFO] Presolve time = 0.00 sec. (0.14 ticks)
[2022-06-28T09:59:17Z, INFO] Tried aggregator 1 time.
[2022-06-28T09:59:17Z, INFO] Reduced MIP has 6 rows, 56 columns, and 306 nonzeros.
[2022-06-28T09:59:17Z, INFO] Reduced MIP has 50 binaries, 6 generals, 0 SOSs, and 0 indicators.
[2022-06-28T09:59:17Z, INFO] Presolve time = 0.00 sec. (0.19 ticks)
[2022-06-28T09:59:17Z, INFO] Represolve time = 0.00 sec. (0.81 ticks)
[2022-06-28T09:59:17Z, INFO]    1518     0        0.0000     7       38.0000      Cuts: 17     3422  100.00%
[2022-06-28T09:59:17Z, INFO]    1518     0        0.0000     8       38.0000      Cuts: 17     3429  100.00%
[2022-06-28T09:59:17Z, INFO]    1518     0        0.0000     7       38.0000      Cuts: 14     3436  100.00%
[2022-06-28T09:59:17Z, INFO]    1518     0        0.0000     7       38.0000      Cuts: 14     3441  100.00%
[2022-06-28T09:59:18Z, INFO]    3918  1669        0.0000     6       38.0000        0.0000     8018  100.00%
[2022-06-28T09:59:18Z, INFO]    6508  2914        0.0000     6       38.0000        0.0000    14256  100.00%
[2022-06-28T09:59:19Z, INFO]    9718  4470        0.0000     6       38.0000        0.0000    22692  100.00%
[2022-06-28T09:59:19Z, INFO] Began writing nodes to disk (directory ./cpxY2cKqj created)
[2022-06-28T09:59:22Z, INFO]   10638  4956       13.6062     6       38.0000        0.0000    25327  100.00%
[2022-06-28T09:59:23Z, INFO]   13478  6155        0.0000     6       38.0000        0.0000    33717  100.00%

Job is completed.
Job has finished with status 'completed'.
dowml> jobs
     #  status      id                                    creation date        type     ver.   size  inputs
     1: completed   cd494377-4843-40a4-ae84-ede7f8c16eda  2022-06-28 11:59:06  cplex    22.1   S     afiro.mps
=>   2: completed   6520e72b-727c-4bfe-adb5-a40d96cf5910  2022-06-28 11:59:13  docplex  22.1   S     markshare.py, markshare1.mps.gz
dowml> dump
Storing 6520e72b-727c-4bfe-adb5-a40d96cf5910/details.json
Storing 6520e72b-727c-4bfe-adb5-a40d96cf5910/markshare.py
Storing 6520e72b-727c-4bfe-adb5-a40d96cf5910/markshare1.mps.gz
Storing 6520e72b-727c-4bfe-adb5-a40d96cf5910/model.lp
Storing 6520e72b-727c-4bfe-adb5-a40d96cf5910/solution.json
Storing 6520e72b-727c-4bfe-adb5-a40d96cf5910/kpis.csv
Storing 6520e72b-727c-4bfe-adb5-a40d96cf5910/stats.csv
Storing 6520e72b-727c-4bfe-adb5-a40d96cf5910/log.txt
dowml> shell ls -l *-*-*-*-*
total 88
-rw-rw-r--  1 nodet  staff  5506 Jun 28 11:59 details.json
-rw-rw-r--  1 nodet  staff    37 Jun 28 11:59 kpis.csv
-rw-rw-r--  1 nodet  staff  7299 Jun 28 11:59 log.txt
-rw-rw-r--  1 nodet  staff   671 Jun 28 11:59 markshare.py
-rw-rw-r--  1 nodet  staff  1607 Jun 28 11:59 markshare1.mps.gz
-rw-rw-r--  1 nodet  staff  4197 Jun 28 11:59 model.lp
-rw-rw-r--  1 nodet  staff  1769 Jun 28 11:59 solution.json
-rw-rw-r--  1 nodet  staff   344 Jun 28 11:59 stats.csv
dowml> delete *
```


## WML credentials

The DOWML client requires some information in order to connect to the Watson
Machine Learning service.  Two pieces of information are required, and the others
are optional.

### Required items

- The `apikey` is a secret that identifies the IBM Cloud user. One typically creates
  one key per application or service, in order to be able to revoke them individually
  if needed.
  To generate such a key, open https://cloud.ibm.com/iam/apikeys, and click the blue
  'Create an IBM Cloud API key' on the right.

- The `url` is the base URL for the REST calls to WML.  The possible values are
  found in https://cloud.ibm.com/apidocs/machine-learning#endpoint-url, 
  and depend on which region you want to use.

- As an alternative to the `url` value, you can use a more user-friendly and 
  easier to remember `region`, with a value that is either `us-south`, `eu-de`,
  `eu-gb` or `jp-tok`.  From this value, _dowml_ will deduce the correct URL to use.
  To avoid ambiguities or duplications, it is not allowed to use both `url` and
  `region`.

### Optional items

Watson Studio and Watson Machine Learning use _spaces_ to group together, and
isolate from each other, the assets that belong to a single project.  These assets 
include the data files submitted, the results of the jobs, and the _deployments_
(software and hardware configurations) that run these jobs.

The DOWML client will connect to the space specified by the user using
either the `--space` command-line argument or the `space_id` item in the credentials.
If neither of these are specified, the client will look for a space named 
_dowml-space_, and will try to create such a space if one doesn't exist.
To create a new space, the DOWML client will need both `cos_resource_crn` and
`ml_instance_crn` to have been specified in the credentials.

- `space_id`: identifier of an existing space to connect to.  Navigate to the 
  'Spaces' tab of your Watson Studio site (e.g. 
  https://eu-de.dataplatform.cloud.ibm.com/ml-runtime/spaces if you are using
  the instance in Germany), right-click on the name of an existing space to
  copy the link. The id of the space is the string of numbers, letters and dashes
  between the last `/` and the `?`.

- `cos_resource_crn`: WML needs to store some data in a Cloud Object Storage 
  instance.  Open
  https://cloud.ibm.com/resources and locate the 'Storage' section.  Create an
  instance of the Cloud Object Storage service if needed. Once it's listed on
  the resource page, click anywhere on the line for that service, except on its
  name.  This will open a pane on the right which lists the CRN.  Click on the
  symbol at the right to copy this information.  This item is required only for 
  the DOWML client to be able to create a space.  If you specified a `space_id`,
  it is not required.
   
- `ml_instance_crn`: similarly, you need to identify an instance of Machine 
  Learning service to use
  to solve your jobs.  In the same page https://cloud.ibm.com/resources, open the
  'Services' section.  The 'Product' columns tells you the type of service.  If
  you don't have a 'Machine Learning' instance already, create one.  Then click
  on the corresponding line anywhere except on the name, and copy the CRN displayed
  in the pane that open on the right.  This item is required only for 
  the DOWML client to be able to create a space.  If you specified a `space_id`,
  it is not required.


## CP4D credentials

The credentials to connect to the WML service _in a (private) CP4D instance_ are 
different from those above that pertain to CP4D _as a service_.  The credentials
look like this:

```
{
   "instance_id": "openshift",
   "version": "4.0",
   "url": "...",
   "username": "...",
   "apikey": "...",
   "space_id": "..."
}
```

- The `url` is the URL of your CP4D instance, with no `/` at the end. 
- The `username` and `apikey` for your user on this cluster.  You can
get the API key in the 'Profile and settings' dialog that's accessible from your
avatar menu in the top-right of the screen.
- `space_id`: the identifier of an existing space to connect to.
The DOWML client will connect to the space specified by the user using
either the `--space` command-line argument or the `space_id` item in the credentials.
If neither of these are specified, the client will look for a space named 
_dowml-space_, and will try to create such a space if one doesn't exist.  On CPD,
unlike for Cloud, no `cos_resource_crn` or `ml_instance_crn` are required.


## Using data assets in Watson Studio

The DOWML library has two modes of operation with respect to sending the models
to the WML service: inline data, or using data assets in Watson Studio.  By default,
data assets are used for inputs, while inline data is used for outputs. 
This can be changed with the `inputs` and `outputs` commands.

With inline data, the model is sent directly to the WML service in the _solve_
request itself, and the output is part of the job details that are downloaded when
asking for information about the job.  
This is the simplest, but it has a number of drawbacks:

- Sending a large model may take a long time, because of network throughput.  Sending
a very large REST request is not at all guaranteed to succeed.  Similarly, if your
job has large outputs, the job may fail while trying to process them.

- When solving several times the same model (e.g. to evaluate different parameters),
the model has to be sent each time.

- In order to display the names of the files that were sent, the _jobs_ command
needs to request this information, and it comes with the content of the files
  themselves.  In other words, every _jobs_ command requires downloading the content
  of all the inline data files for all the jobs that exist in the space.

Using data assets in Watson Studio as an intermediate step alleviate all these issues:

- Once the model has been uploaded to Watson Studio, it will be reused for
subsequent jobs without the need to upload it again.

- The job requests refer to the files indirectly, via URLs.  Therefore, they don't
take much space, and listing the jobs doesn't imply to download the content of the
  files.

- Uploading to Watson Studio is done through specialized code that doesn't just send a single
request.  Rather, it divides the upload in multiple reasonably sized chunks that each
  are uploaded individually, with restart if necessary.  Uploading big files is
  therefore much less prone to failure.

