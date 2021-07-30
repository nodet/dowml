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

    def test_simple_cplex_inline(self):
        l = self.lib
        l.inline = True
        id = l.solve('examples/afiro.mps')
        l.wait_for_job_end(id)
        details = l.get_job_details(id)
        self.assertEqual(details['entity']['decision_optimization']['status']['state'], 'completed')
        l.delete_job(id, hard=True)
        self.assertNotIn(id, [j.id for j in l.get_jobs()])


if __name__ == '__main__':
    main()
