import argparse
import os
import tempfile
import statistics
import time
from datetime import datetime

import requests

from dowml.lib import DOWMLLib, _CredentialsProvider

NB_MODELS = 20
QUANTILE = 10


# We will patch the 'requests' function that's used by APIClient
# So first we save the original function
orig_requests_session_send = requests.Session.send
# Counters for requests statistics
total_requests = 0
warn_requests = 0


# And here's the function that replaces 'Session.send'
def patched_requests_session_send(*arguments, **kwargs):
    global total_requests, warn_requests

    session, prepared_request = arguments
    method = prepared_request.method
    url = prepared_request.url
    dt = datetime.now()
    iso = dt.isoformat(sep=' ', timespec='milliseconds')
    print(f'{iso} {method} {url}')
    resp = orig_requests_session_send(*arguments, **kwargs)
    total_requests += 1
    dt2 = datetime.now()
    diff = (dt2 - dt).total_seconds()
    warn = ''
    if diff > 10.0:
        warn = f' <== WARNING! This one took {diff} seconds'
        warn_requests += 1
    iso = dt2.isoformat(sep=' ', timespec='milliseconds')
    print(f'{iso} {resp.status_code} {warn}')
    return resp


def diff_time(t1_str, t2_str):
    """"Returns t2 - t1 with the appropriate gymnastics"""
    t1 = datetime.fromisoformat(t1_str[:-1])
    t2 = datetime.fromisoformat(t2_str[:-1])
    return (t2 - t1).total_seconds()


def run_one_model(lib, path):
    dt = datetime.now()
    job_id = lib.solve(path)
    submit_time = (datetime.now() - dt).total_seconds()
    _, details = lib.wait_for_job_end(job_id)
    # It happens very often that the WS job is still 'in progress' when the WML job
    # has just completed, and deleting it fails.  Waiting a little bit, to give the
    # platform enough time to update the WS job should help.
    time.sleep(2)
    lib.delete_job(job_id, True)

    queued_time = diff_time(details['metadata']['created_at'],
                            details['entity']['decision_optimization']['status']['running_at'])
    stored_time = diff_time(details['entity']['decision_optimization']['status']['completed_at'],
                            details['metadata']['modified_at'])
    print(f'Job {job_id} was submitted in {submit_time} seconds, '
          f'queued for {queued_time} seconds, '
          f'and stored after {stored_time} seconds.')
    return submit_time + queued_time + stored_time


def test_one_region(number, wml_cred_file=None, space_id=None, url=None, region=None):
    # logging.basicConfig(force=True, format='%(asctime)s %(message)s')
    # logging.getLogger(dowml.dowmllib.DOWMLLib.__name__).setLevel(logging.DEBUG)
    requests.Session.send = patched_requests_session_send
    global total_requests, warn_requests
    total_requests = 0
    warn_requests = 0

    try:
        lib = DOWMLLib(wml_cred_file, space_id, url=url, region=region)
        print(f'Using URL: {lib.url}')
        lib.model_type = 'docplex'
        lib.inputs = 'inline'
        lib.outputs = 'inline'

        handle, path = tempfile.mkstemp(suffix='.py', text=True)
        try:
            os.write(handle, b"print('Running the code...')")
        finally:
            os.close(handle)
        try:
            print('Running a first job to create/warm up the deployment...')
            run_one_model(lib, path)
            print('Now, we start counting...')

            times = [run_one_model(lib, path) for i in range(number)]
            print(times)
            print('Mean: ', statistics.mean(times))
            print('Median: ', statistics.median(times))
            print(f'{QUANTILE - 1}-th quantile: ', statistics.quantiles(times, n=QUANTILE)[QUANTILE - 2])

            print(f'warning/total # of requests: {warn_requests}/{total_requests}')
        finally:
            os.remove(path)
    finally:
        requests.Session.send = orig_requests_session_send


DESCRIPTION = \
    '''Checking WML job submission, queuing and saving speed.'''


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wml-cred-file', '-w', default=None,
                        help=f'Name of the file from which to read WML '
                             f'credentials. If not specified, credentials '
                             f'are read from environment variable '
                             f'${_CredentialsProvider.ENVIRONMENT_VARIABLE_NAME}. If '
                             f'no such variable exists, but variable '
                             f'${_CredentialsProvider.ENVIRONMENT_VARIABLE_NAME_FILE} '
                             f'exists, tries to read that file.')
    parser.add_argument('--space', '-s', default=None,
                        help=f'Id of the space to connect to. Takes precedence over '
                             f'the one specified in the credentials under the '
                             f'\'{_CredentialsProvider.SPACE_ID}\' key, if any.')
    parser.add_argument('--url', '-u', default=None,
                        help=f'URL to use for the Machine Learning service. Takes precedence over '
                             f'the one specified in the credentials under the '
                             f'\'{_CredentialsProvider.URL}\' key, if any. '
                             f'Incompatible with --region argument.')
    regions = list(_CredentialsProvider.REGION_TO_URL.keys())
    parser.add_argument('--region', '-r', default=None,
                        help=f'Region to use for the Machine Learning service. Takes precedence over '
                             f'the region or URL specified in the credentials, if any. '
                             f'Incompatible with --url argument. '
                             f'Possible values for the region are {regions}.')
    parser.add_argument('--number', '-n', default=NB_MODELS,
                        help='Number of jobs to submit.')
    args = parser.parse_args()
    test_one_region(int(args.number), wml_cred_file=args.wml_cred_file,
                    space_id=args.space, url=args.url, region=args.region)


if __name__ == '__main__':
    main()
