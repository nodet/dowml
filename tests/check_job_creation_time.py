import os
import tempfile

import dowml
import statistics
from datetime import datetime

from dowml.lib import DOWMLLib

NB_MODELS = 20
QUANTILE = 10


def diff_time(t1_str, t2_str):
    """"Returns t2 - t1 with the appropriate gymnastics"""
    t1 = datetime.fromisoformat(t1_str[:-1])
    t2 = datetime.fromisoformat(t2_str[:-1])
    return (t2 - t1).total_seconds()


def run_one_model(lib, path):
    dt = datetime.now()
    job_id = lib.solve(path)
    submit_time = (datetime.now() - dt).total_seconds()
    lib.wait_for_job_end(job_id)
    lib.delete_job(job_id, True)

    # queued_time = diff_time(details['metadata']['created_at'],
    #                        details['entity']['decision_optimization']['status']['running_at'])
    # stored_time = diff_time(details['entity']['decision_optimization']['status']['completed_at'],
    #                         details['metadata']['modified_at'])
    print(f'Job {job_id} was submitted in {submit_time} seconds.')
    return submit_time


def main():
    lib = DOWMLLib()
    url = lib._wml_credentials['url']
    print(f'Using URL: {url}')
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

        times = [run_one_model(lib, path) for i in range(NB_MODELS)]
        print(times)
        print('Mean: ', statistics.mean(times))
        print('Median: ', statistics.median(times))
        print(f'{QUANTILE - 1}-th quantile: ', statistics.quantiles(times, n=QUANTILE)[QUANTILE - 2])
    finally:
        os.remove(path)


if __name__ == '__main__':
    main()
