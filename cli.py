import argparse
import logging
import pprint
from cmd import Cmd
from dowmlclient import DOWMLClient


class DOWMLInteractive(Cmd):
    prompt = 'dowml> '
    intro = ('Decision Optimization in WML Interactive.\n'
             'Type ? for a list of commands')

    def __init__(self, wml_cred_file):
        super().__init__()
        self.client = DOWMLClient(wml_cred_file)

    def do_exit(self, inp):
        '''Exit the Interactive'''
        return True

    def do_solve(self, path):
        '''Start a solve job of the CPLEX model in the specified file'''
        job_id = self.client.solve(path, False)
        print(f'Job id: {job_id}')

    def do_wait(self, job_id):
        '''Wait until the job is finished, printing activity'''
        self.client.wait_for_job_end(job_id, True)

    def do_jobs(self, _):
        '''List all the jobs in this deployment'''
        jobs = self.client.get_jobs()
        for i, j in enumerate(jobs, start=1):
            status, id = j
            print(f'{i:>3}: {status:>10}  {id}')

    def do_log(self, job_id):
        '''Print the CPLEX log for the given job'''
        log = self.client.get_log(job_id)
        print(log)

    def do_details(self, job_id):
        '''Print most of the details for the given job'''
        details = self.client.get_job_details(job_id)
        pprint.pprint(details, indent=4, width=120)

    def do_delete(self, job_id):
        '''Delete the job with the given id'''
        self.client.delete_job(job_id, True)

    def do_cancel(self, job_id):
        '''Stops the job with the given id'''
        self.client.delete_job(job_id, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive program for DO on WML')
    parser.add_argument('--wml-cred-file', default=None,
                        help='Name of the file from which to read WML credentials. '
                             'If not specified, credentials are read from an environment variable')
    args = parser.parse_args()

    logging.basicConfig(force=True, format='%(asctime)s %(message)s', level=logging.INFO)
    DOWMLInteractive(args.wml_cred_file).cmdloop()
