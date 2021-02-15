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
cancel  delete  details  exit  help  jobs  log  output size  solve  type  wait

dowml> help cancel
Stops the job with the given id.
dowml> help delete
Delete the job with the given id. Use '*' to delete all the jobs.
dowml> help details
Print most of the details for the given job. Add 'full' to get the contents.
dowml> help exit
Exit this program.
dowml> help help
List available commands with "help" or detailed help with "help cmd".
dowml> help jobs
List all the jobs in the space.
dowml> help log
Print the engine log for the given job.
dowml> help output
Download all the outputs of a job
dowml> help size
Print current deployment size (if no argument), or set the deployment size.
dowml> help solve
Start a job to solve a CPLEX model. At least one file of the correct type must be specified as argument.
dowml> help type
Print current model type (if no argument), or set the model type.
dowml> help wait
Wait until the job is finished, printing activity. Hit Ctrl-C to interrupt.
dowml> jobs
2021-02-14 14:56:37,141 Creating the connexion...
2021-02-14 14:56:41,990 Creating the connexion succeeded.  Client version is 1.0.45
2021-02-14 14:56:41,990 Fetching existing spaces...
2021-02-14 14:56:43,352 Got the list. Looking for space named 'DOWMLClient-space'
2021-02-14 14:56:43,352 Found it.
2021-02-14 14:56:43,352 Space id: 15f1a4b1-1e2b-4a60-8b4f-cf540ca65d36
2021-02-14 14:56:43,352 Setting default space...
2021-02-14 14:56:52,982 Done.
     #   status     id                                    creation date             inputs
     1:     failed  13c55254-a56d-479d-acb9-93d1673fd421  2021-02-13T08:40:29.458Z  my_model.py, markshare1.mps.gz
     2:     failed  2f48b138-f7d1-4812-94b5-f7ccdde3a242  2021-02-13T11:50:20.705Z  afiro.mps
dowml> details 1
2021-02-14 14:57:31,821 Fetching output...
2021-02-14 14:57:33,147 Done.
{   'entity': {   'decision_optimization': {   'input_data': [   {'content': '[not shown]', 'id': 'my_model.py'},
                                                                 {'content': '[not shown]', 'id': 'markshare1.mps.gz'}],
                                               'output_data': [{'id': '.*\\.json'}, {'id': '.*\\.txt'}],
                                               'solve_parameters': {   'oaas.includeInputData': 'false',
                                                                       'oaas.logAttachmentName': 'log.txt',
                                                                       'oaas.logTailEnabled': 'true',
                                                                       'oaas.resultFormat': 'JSON'},
                                               'status': {   'completed_at': '2021-02-13T08:40:29.946Z',
                                                             'failure': {   'errors': [   {   'code': 'instance_quota_exceeded',
                                                                                              'message': 'This job '
                                                                                                         'cannot be '
                                                                                                         'processed '
                                                                                                         'because it '
                                                                                                         'exceeds the '
                                                                                                         'allocated '
                                                                                                         'capacity '
                                                                                                         'unit hours '
                                                                                                         '(CUH). '
                                                                                                         'Increase the '
                                                                                                         'compute '
                                                                                                         'resources '
                                                                                                         'for this job '
                                                                                                         'and try '
                                                                                                         'again.'}],
                                                                            'trace': '5c5f2ae9ec0bf63e2154854e3eb18284'},
                                                             'state': 'failed'}},
                  'deployment': {'id': '38ad90bc-e3c3-4e96-b284-f86f3ea7fe73'},
                  'platform_job': {   'job_id': '2afec797-73f3-40c8-83e2-6889f906c9f7',
                                      'run_id': '5c25b9ec-53eb-497e-9d8a-e9638fb4c9d8'}},
    'metadata': {   'created_at': '2021-02-13T08:40:29.458Z',
                    'id': '13c55254-a56d-479d-acb9-93d1673fd421',
                    'name': 'name_08dc32b6-cebf-49ad-82bd-7a8079ee1aa4',
                    'space_id': '15f1a4b1-1e2b-4a60-8b4f-cf540ca65d36'}}
dowml>
```
