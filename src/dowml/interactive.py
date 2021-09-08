#  --------------------------------------------------------------------------
#  Source file provided under Apache License, Version 2.0, January 2004,
#  http://www.apache.org/licenses/
#  (c) Copyright IBM Corp. 2021
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  --------------------------------------------------------------------------

import argparse
import logging
import os
import pprint
import re
import sys

import requests

from cmd import Cmd
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure

import dowml
from dowml.dowmllib import DOWMLLib, InvalidCredentials, NoCredentialsToCreateSpace, _CredentialsProvider


class CommandNeedsJobID(Exception):
    pass


class CommandNeedsNonNullInteger(Exception):
    pass


class InvalidArgumentForCommand(Exception):
    def __init__(self, command, value, possible_values):
        super().__init__()
        self.possible_values = possible_values
        self.command = command
        self.value = value

    pass


INTERACTIVE_BANNER = \
    f'''Decision Optimization in WML Interactive, version {dowml.VERSION}.
Submit and manage Decision Optimization models interactively.
(c) Copyright IBM Corp. 2021'''


class DOWMLInteractive(Cmd):
    prompt = 'dowml> '
    intro = (f'''
{INTERACTIVE_BANNER}

Type ? for a list of commands.

Most commands need an argument that can be either a job id, or the number
of the job, as displayed by the 'jobs' command.  If a command requires a
job id, but none is specified, the last one is used.

''')

    def __init__(self, wml_cred_file, space_id=None):
        super().__init__()
        self.lib = DOWMLLib(wml_cred_file, space_id)
        self.jobs = []
        self.last_job_id = None

    def emptyline(self) -> bool:
        # Just hitting enter should _not_ repeat a command
        return False

    def _get_and_remember_job_id(self, number):
        self.last_job_id = self._number_to_id(number)
        return self.last_job_id

    def _number_to_id(self, number):
        if not number:
            # If nothing specified, use the last job id
            number = self.last_job_id
        if not number:
            raise CommandNeedsJobID
        if number in self.jobs:
            # Easy: we simply have an existing job id
            return number
        if str.isdigit(number):
            num = int(number)
            if 0 < num <= len(self.jobs):
                return self.jobs[num - 1]
        # It may be a job we don't know yet...
        return number

    @staticmethod
    def do_exit(_):
        """Exit this program."""
        return True

    def print_known_types(self):
        known_types = ', '.join(self.lib.MODEL_TYPES)
        print(f'Known types: {known_types}.')

    def do_type(self, model_type):
        """type [type-of-model]
Prints current model type and known types (if no argument), or sets the model type."""
        if not model_type:
            print(f'Current model type: {self.lib.model_type}.')
            self.print_known_types()
            # Let's make sure we don't set the model_type to None, but
            # return immediately
            return
        if model_type not in self.lib.MODEL_TYPES:
            print(f'Warning: unknown model type \'{model_type}\'.')
            self.print_known_types()
        # We set the type nevertheless: this code may not be up-to-date
        self.lib.model_type = model_type

    def print_known_sizes(self):
        known_sizes = ', '.join(self.lib.TSHIRT_SIZES)
        print(f'Known sizes: {known_sizes}.')

    def do_size(self, tshirt_size):
        """size [size-of-deployment]
Prints current deployment size and known sizes (if no argument), or sets the deployment size."""
        if not tshirt_size:
            print(f'Current size: {self.lib.tshirt_size}.')
            self.print_known_sizes()
            # Let's make sure we don't set the tshirt_size to None, but
            # return immediately
            return
        if tshirt_size not in self.lib.TSHIRT_SIZES:
            print(f'Warning: unknown tee-shirt size \'{tshirt_size}\'.')
            self.print_known_sizes()
        # We set the size nevertheless: this code may not be up-to-date
        self.lib.tshirt_size = tshirt_size

    def do_time(self, limit):
        """time [non-negative-integer]
Prints current time limit (if no argument), or sets the time limit (in seconds). 0 to remove the time limit."""
        if not limit:
            print(f'Current time limit: {self.lib.timelimit}')
            return
        try:
            limit = int(limit)
        except ValueError:
            raise CommandNeedsNonNullInteger
        if limit < 0:
            raise CommandNeedsNonNullInteger
        if limit == 0:
            limit = None
        self.lib.timelimit = limit

    def do_version(self, version):
        """version [string]
Prints default DO version (if no argument) and available ones, or sets the DO version to the specified string."""
        if not version:
            print(f'Decision Optimization version: {self.lib.do_version}')
            versions = ', '.join(self.lib.get_available_versions())
            print(f'Available versions: {versions}')
            return
        self.lib.do_version = version

    def do_inline(self, arg):
        """inline [yes|no]   --  DEPRECATED: use 'inputs' instead
Prints whether jobs are created with inline data (if no argument), or sets the
flag ('yes' or 'no').

In 'inline no' mode, which is the default, jobs use data assets stored on the
platform for their inputs, rather than uploading the content of files as part of
each job payload.  If a file name (stripped from its path) matches the name of
an existing data asset, the content of the file is not uploaded (actually, the
file may not even exist locally) and the content of that data asset is used
instead. If a file name is prefixed with a '+', an existing data asset will be
replaced with the content of the local file.

In 'inline yes' mode, the content of the specified files is always uploaded to
the platform as part of the job payload, and not stored as a data asset."""
        if not arg:
            flag = 'yes' if self.lib.inline else 'no'
            print(f'inline: {flag}')
            return
        if arg == 'yes':
            self.lib.inputs = 'inline'
        elif arg == 'no':
            self.lib.inputs = 'assets'
        else:
            raise InvalidArgumentForCommand('inline', arg, ['yes', 'no'])

    def do_inputs(self, arg):
        """inputs [inline|assets]
Displays (if no argument) or sets (if given an argument) the type of inputs
that jobs will use.  Possible values are 'inline' and 'assets'.

In 'inputs inline' mode, jobs use inline input data: the files that constitute
the job are included in the job creation request.

In 'inputs assets' mode, which is the default, jobs use data assets stored on the
platform for their inputs, rather than uploading the content of files as part of
each job payload.  If a file name (stripped from its path) matches the name of
an existing data asset, the content of the file is not uploaded (actually, it's
ok if the file does not even exist locally) and the content of that data asset is
used instead. If a file name is prefixed with a '+', the file is always uploaded
and an existing data asset is replaced with the content of the local file."""
        if not arg:
            print(f'inputs: {self.lib.inputs}')
            return
        allowed_values = ['inline', 'assets']
        if arg in allowed_values:
            self.lib.inputs = arg
        else:
            raise InvalidArgumentForCommand('inputs', arg, allowed_values)

    def do_outputs(self, arg):
        """outputs [inline|assets]
Displays (if no argument) or sets (if given an argument) the type of outputs
that jobs will use.  Possible values are 'inline' and 'assets'.

In 'outputs inline' mode, which is the default, jobs use inline output data: the
results of the job are included in the job details. This is fine for most jobs.

In 'outputs assets' mode, the job outputs are stored as data assets in the job's
deployment space.  Their names start with the id of the job, followed by the
file's name. This is useful for jobs that create very large outputs."""
        if not arg:
            print(f'outputs: {self.lib.outputs}')
            return
        allowed_values = ['inline', 'assets']
        if arg in allowed_values:
            self.lib.outputs = arg
        else:
            raise InvalidArgumentForCommand('outputs', arg, allowed_values)

    def do_solve(self, paths):
        """solve file1 [file2 ... [filen]]
Starts a job to solve a DO model. At least one file with the correct extension
for the current job type must be specified as argument.  In 'inline no' mode,
if the file name is prefixed with '+', the file is uploaded even if an asset
with that name already exists.  In 'inline yes' mode, a '+' prefix has no effect.
Wildcards (* and ?) are accepted (but no ~)."""
        if not paths:
            print('This command requires at least one file name as argument.')
            return
        job_id = None
        try:
            job_id = self.lib.solve(paths)
        except FileNotFoundError as exception:
            print(exception)
        else:
            print(f'Job id: {job_id}')
        self.last_job_id = job_id

    def do_wait(self, job_id):
        """wait [job]
Waits until the job is finished, printing activity. Hit Ctrl-C to interrupt.
job is either a job number or a job id. Uses current job if not specified."""
        job_id = self._get_and_remember_job_id(job_id)
        self.lib.wait_for_job_end(job_id, True)

    def _cache_jobs(self):
        jobs = self.lib.get_jobs()
        self.jobs = [j.id for j in jobs]
        if len(jobs) == 1:
            self.last_job_id = self.jobs[0]
        return jobs

    def do_jobs(self, _):
        """jobs
Lists all the jobs in the space.
Current job, if any, is indicated with an arrow."""
        jobs = self._cache_jobs()
        print('     #  status      id                                    '
              'creation date        type     ver.   size  inputs')
        for i, j in enumerate(jobs, start=1):
            # Prepare list of input files
            names = ', '.join(j.names)
            # Mark the job used if none specified
            mark = '   '
            if j.id == self.last_job_id:
                mark = '=> '
            print(f'{mark}{i:>3}: {j.status:<10}  {j.id}  '
                  f'{j.created}  {j.type:<7}  {j.version:<5}  {j.size:<4}  {names}')

    def do_log(self, job_id):
        """log [job]
Prints the engine log for the given job.
job is either a job number or a job id. Uses current job if not specified."""
        job_id = self._get_and_remember_job_id(job_id)
        log = self.lib.get_log(job_id)
        print(log)

    def do_output(self, job_id):
        """output [job]
Downloads all the outputs of a job, as well as the details/status of the job.
job is either a job number or a job id. Uses current job if not specified."""
        job_id = self._get_and_remember_job_id(job_id)
        details = self.lib.get_job_details(job_id, with_contents='full')
        self._output_regular_files(job_id, details)
        # We don't want to store all the outputs in the details themselves
        self.lib.filter_large_chunks_from_details(details)
        details = pprint.pformat(details)
        self.save_content(job_id, 'details.json', details, text=True)

    def _output_regular_files(self, job_id, details):
        outputs = self.lib.get_output(details, tabular_as_csv=True)
        for name in outputs:
            self.save_content(job_id, name, outputs[name])

    @staticmethod
    def save_content(job_id, name, content, text=False):
        try:
            os.mkdir(job_id)
        except FileExistsError:
            pass
        file_name = f'{job_id}/{name}'
        flags = 'wt' if text else 'wb'
        with open(file_name, flags) as f:
            print(f'Storing {file_name}')
            f.write(content)

    def do_details(self, arguments, printer=pprint.pprint):
        """details [job] [names|full|log]
Prints most of the details for the given job. Add 'names' to get the names of the
input and output file names. Add 'full' to get the actual contents of inputs and outputs.
job is either a job number or a job id. Uses current job if not specified."""
        with_contents = False
        job_id = None
        for arg in arguments.split():
            if arg == 'full' or arg == 'names':
                with_contents = arg
            else:
                job_id = self._get_and_remember_job_id(arg)
        job_id = self._get_and_remember_job_id(job_id)
        details = self.lib.get_job_details(job_id, with_contents=with_contents)
        printer(details, indent=4, width=120)

    def do_delete(self, job_id):
        """delete [job|*|n-m]
Deletes the job specified. 'job' is either a job number or a job id.
Use '*' to delete all the jobs. A range of jobs, specified by their numbers,
can be deleted using 'n-m'.
Uses current job if not specified."""
        if job_id == '*':
            # Update the list of jobs before trying to delete them
            self._cache_jobs()
            while self.jobs:
                job = self.jobs[0]
                self.delete_one_job(job)
        elif m := re.fullmatch('(\\d+)-(\\d+)', job_id):
            n1 = int(m.group(1))
            n2 = int(m.group(2))
            # We must gather the list of all the ids of jobs we want to delete
            # right from the start, because deleting each job will update the
            # list while iterating
            ids = self.jobs[n1-1:n2]
            for i in ids:
                self.delete_one_job(i)
        else:
            self.delete_one_job(job_id)

    def delete_one_job(self, job_id):
        previous_job = self.last_job_id
        job_id = self._get_and_remember_job_id(job_id)
        self.lib.delete_job(job_id, True)
        if job_id in self.jobs:
            self.jobs.remove(job_id)
        assert job_id not in self.jobs  # Because a job appears only once
        if previous_job != job_id:
            # The current job was a job that's still there.
            # Let's keep that one as current job
            self.last_job_id = previous_job
        else:
            # The job that was deleted was the current one
            # Generally speaking, we can't decide of a new current job...
            self.last_job_id = None
        if len(self.jobs) == 1:
            # ... except if there's only one!
            self.last_job_id = self.jobs[0]

    def do_cancel(self, job_id):
        """cancel [job]
Stops the job specified.
job is either a job number or a job id. Uses current job if not specified."""
        job_id = self._get_and_remember_job_id(job_id)
        self.lib.delete_job(job_id, False)

    @staticmethod
    def do_shell(command):
        """shell command
Runs the specified command in a shell."""
        os.system(command)
        # We ignore the value returned as anything non-zero would just stop
        # the Interactive.
        return 0


def main_loop(wml_cred_file, space_id, commands, prompt_at_the_end):
    instance = DOWMLInteractive(wml_cred_file, space_id)
    # By default, we want to run the command loop
    loop = True
    for c in commands:
        if loop:
            # This is the first command we carry out. Let's print intro
            print(instance.intro, end='')
            # And make sure we won't print it again
            instance.intro = ''
        # We run the command loop iff this was asked for
        loop = prompt_at_the_end
        print(f'{instance.prompt}{c}')
        instance.onecmd(c)
    while loop:
        # Generally speaking, one command loop is all we need, and we
        # should not run another, except in case of errors
        loop = False
        try:
            instance.cmdloop()
        except KeyboardInterrupt:
            # The user interrupted. That's perfectly fine...
            loop = True
        except ApiRequestFailure:
            # This happens when an invalid job id is specified. We want
            # to keep running.
            loop = True
        except requests.exceptions.ConnectionError as e:
            print(e)
            loop = True
        except CommandNeedsJobID:
            print('This command requires a jod id or number, but you '
                  'didn\'t specify one.  And there is no current job either.')
            loop = True
        except InvalidArgumentForCommand as e:
            print(f'Invalid argument \'{e.value}\' for command \'{e.command}\'. '
                  f'Possible values are {e.possible_values}. '
                  f'Please type \'help {e.command}\' to learn more.')
            loop = True
        except NoCredentialsToCreateSpace as e:
            print(e)
            # There's nothing the user could do at this point: no need to loop
        finally:
            # But let's not print again the starting banner
            instance.intro = ''


# We will mock the 'requests' function that's used by APIClient
# So first we save the original function
orig_requests_session_send = requests.Session.send


# And here's the function that replaces 'Session.send'
def mocked_requests_session_send(*arguments, **kwargs):
    session, prepared_request = arguments
    method = prepared_request.method
    url = prepared_request.url
    #       2021-02-17 16:59:39,710
    print(f'           {method} {url}')
    resp = orig_requests_session_send(*arguments, **kwargs)
    # print(f'                        {resp.status_code}')
    return resp


def interactive():
    parser = argparse.ArgumentParser(description=INTERACTIVE_BANNER,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wml-cred-file', '-w', default=None,
                        help=f'Name of the file from which to read WML '
                             f'credentials. If not specified, credentials '
                             f'are read from environment variable '
                             f'${_CredentialsProvider.ENVIRONMENT_VARIABLE_NAME}. If '
                             f'no such variable exists, but variable '
                             f'${_CredentialsProvider.ENVIRONMENT_VARIABLE_NAME_FILE} '
                             f'exists, tries to read that file.')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Verbose mode.  Causes the program to print debugging '
                             'messages about its progress.  Multiple -v options '
                             'increase the verbosity.  The maximum is 4.')
    parser.add_argument('--commands', '-c', nargs='*', default=[],
                        help='Carries out the specified commands.  Each command '
                             'is executed as if it had been specified at the prompt. '
                             'The program stops after last command, unless --input '
                             'is used.')
    parser.add_argument('--input', '-i', action='store_true',
                        help='Prompts for new input commands even if some commands '
                             'have been specified as arguments using --commands.')
    parser.add_argument('--space', '-s', default=None,
                        help=f'Id of the space to connect to. Takes precedence over '
                             f'the one specified in the credentials under the '
                             f'\'{_CredentialsProvider.SPACE_ID}\' key, if any.')
    args = parser.parse_args()

    # Last logging level repeated as many times as necessary to accommodate
    # for levels higher than what logging does (eg REST queries)
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG, logging.DEBUG, logging.DEBUG]
    # The force parameter is not listed in the arguments to basicConfig
    # noinspection PyArgumentList
    logging.basicConfig(force=True, format='%(asctime)s %(message)s')
    logging.getLogger(DOWMLLib.__name__).setLevel(log_levels[args.verbose])
    logging.getLogger(_CredentialsProvider.__name__).setLevel(log_levels[args.verbose])

    if args.verbose >= 3:
        # Let's report the sending of REST queries
        requests.Session.send = mocked_requests_session_send

    if args.verbose >= 4:
        # Let's print the responses we get
        logging.getLogger('ibm_watson_machine_learning').setLevel(logging.DEBUG)
        # logging.getLogger('urllib3').setLevel(logging.DEBUG)
        # logging.getLogger('requests').setLevel(logging.DEBUG)
        # logging.getLogger('swagger_client').setLevel(logging.DEBUG)
        # logging.getLogger('ibm_botocore').setLevel(logging.DEBUG)
        # logging.getLogger('ibm_boto3').setLevel(logging.DEBUG)

    try:
        main_loop(args.wml_cred_file, args.space, args.commands, args.input)
    except InvalidCredentials:
        print('\nERROR: credentials not found!\n')
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    interactive()
