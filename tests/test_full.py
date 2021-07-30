import pprint

from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure

from dowmllib import DOWMLLib
from unittest import TestCase, main


class TestRunningOnWML(TestCase):

    def setUp(self) -> None:
        # Getting the credentials from the environment, so that this can run both
        # locally and in GitHub
        lib = DOWMLLib()
        self.lib = lib

    def job_solves_to_completion(self, paths, type=None, inline=None):
        l = self.lib
        if inline is not None:
            l.inline = inline
        if type is not None:
            l.model_type = type
        id = l.solve(paths)
        l.wait_for_job_end(id)
        details = l.get_job_details(id)
        self.assertEqual(details['entity']['decision_optimization']['status']['state'], 'completed')
        l.delete_job(id, hard=True)
        self.assertNotIn(id, [j.id for j in l.get_jobs()])

    def test_simple_cplex_inline(self):
        self.job_solves_to_completion(inline=True, paths='examples/afiro.mps')

    def test_docplex_inline(self):
        # 'type docplex' 'solve examples/markshare.py examples/markshare1.mps.gz' wait jobs output 'details full' delete
        self.job_solves_to_completion(inline=True, type='docplex', paths='examples/markshare.py examples/markshare1.mps.gz')

if __name__ == '__main__':
    main()
