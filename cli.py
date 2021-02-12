import argparse
import logging
import pprint
import requests

from cmd import Cmd
from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure

from dowmlclient import DOWMLClient, InvalidCredentials


class CommandNeedsJobID(Exception):
    pass


class DOWMLInteractive(Cmd):
    prompt = 'dowml> '
    intro = ('''
Decision Optimization in WML Interactive.
Submit and manage CPLEX models interactively.

Type ? for a list of commands.

Most commands need an argument that can be either a job id, or the number
of the job, as displayed by the 'jobs' command.  If a command requires a
job id, but none is specified, the last one is used.
''')

    def __init__(self, wml_cred_file):
        super().__init__()
        self.client = DOWMLClient(wml_cred_file)
        self.jobs = []
        self.last_job_id = None

    def emptyline(self) -> bool:
        # Just hitting enter should _not_ repeat a command
        return False

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

    def do_exit(self, inp):
        '''Exit this program.'''
        return True

    def do_type(self, model_type):
        '''Print current model type (if no argument), or set the model type.'''
        known_types = ', '.join(self.client.MODEL_TYPES)
        if not model_type:
            print(f'Current model type: {self.client.model_type}')
            # Let's make sure we don't set the model_type to None, but
            # return immediately
            return
        if model_type not in self.client.MODEL_TYPES:
            print(f'Warning: unknown model type \'{model_type}\'. Known types: {known_types}')
        # We set the type nevertheless: this code may not be up-to-date
        self.client.model_type = model_type

    def do_solve(self, paths):
        '''Start a job to solve a CPLEX model. At least one file of the correct type must be specified as argument.'''
        if not paths:
            print('This command requires at least one file name as argument.')
            return
        job_id = None
        try:
            job_id = self.client.solve(paths, False)
        except FileNotFoundError as e:
            print(e)
        else:
            print(f'Job id: {job_id}')
        self.last_job_id = job_id

    def do_wait(self, job_id):
        '''Wait until the job is finished, printing activity. Hit Ctrl-C to interrupt.'''
        job_id = self._number_to_id(job_id)
        try:
            self.client.wait_for_job_end(job_id, True)
        except KeyboardInterrupt:
            # The user interrupted. That's perfectly fine...
            pass
        self.last_job_id = job_id

    def do_jobs(self, _):
        '''List all the jobs in the space.'''
        jobs = self.client.get_jobs()
        self.jobs = []
        print('     #   status     id                                    creation date             inputs')
        for i, j in enumerate(jobs, start=1):
            # Prepare list of input files
            names = ', '.join(j.names)
            # Add this job id in the list, to allow for translation from job number
            self.jobs.append(j.id)
            # Mark the job used if none specified
            mark = '   '
            if j.id == self.last_job_id:
                mark = '=> '
            print(f'{mark}{i:>3}: {j.status:>10}  {j.id}  {j.created}  {names}')

    def do_log(self, job_id):
        '''Print the engine log for the given job.'''
        job_id = self._number_to_id(job_id)
        log = self.client.get_log(job_id)
        print(log)
        self.last_job_id = job_id

    def do_details(self, job_id):
        '''Print most of the details for the given job.'''
        job_id = self._number_to_id(job_id)
        details = self.client.get_job_details(job_id)
        pprint.pprint(details, indent=4, width=120)
        self.last_job_id = job_id

    def do_delete(self, job_id):
        '''Delete the job with the given id.'''
        job_id = self._number_to_id(job_id)
        self.client.delete_job(job_id, True)
        self.last_job_id = None

    def do_cancel(self, job_id):
        '''Stops the job with the given id.'''
        job_id = self._number_to_id(job_id)
        self.client.delete_job(job_id, False)
        self.last_job_id = job_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive program for DO on WML')
    parser.add_argument('--wml-cred-file', default=None,
                        help='Name of the file from which to read WML credentials. '
                             'If not specified, credentials are read from an environment variable')
    args = parser.parse_args()

    logging.basicConfig(force=True, format='%(asctime)s %(message)s', level=logging.INFO)
    try:
        dowml = DOWMLInteractive(args.wml_cred_file)
        while True:
            again = False
            try:
                dowml.cmdloop()
            except ApiRequestFailure as failure:
                # This happens when an invalid job id is specified. We want
                # to keep running.
                again = True
            except requests.exceptions.ConnectionError as e:
                print(e)
                again = True
            except CommandNeedsJobID:
                print(f'This command requires a jod id or number.')
                again = True
            finally:
                # But let's not print again the starting banner
                dowml.intro = ''
            if not again:
                break
    except InvalidCredentials:
        print(f'\nERROR: credentials not found!\n')
        parser.print_help()