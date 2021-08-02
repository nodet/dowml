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
        self.job_solves_to_completion(inline=True,
                                      type='docplex',
                                      paths='examples/markshare.py examples/markshare1.mps.gz')


class TestDetailsAndOutputs(TestCase):
    """"This class runs two CPLEX models, one inline, the other by reference.
    And it checks that the details and outputs are as expected"""

    def assertLogIsMentionedButNoContent(self, details):
        self.assertOutputIsMentionedButNoContent(details, 'log.txt')

    def assertSolutionIsMentionedButNoContent(self, details):
        self.assertOutputIsMentionedButNoContent(details, 'solution.json')

    def assertOutputIsMentionedButNoContent(self, details, name):
        output = details['entity']['decision_optimization']['output_data']
        seen_output = False
        for o in output:
            if o['id'] == name:
                self.assertFalse(seen_output)
                seen_output = True
                content = o['content']
                self.assertEqual(content, '[not shown]')

    @classmethod
    def setUpClass(cls) -> None:
        l = DOWMLLib()
        id_not_inline = l.solve('examples/afiro.mps')
        l.inline = True
        id_inline = l.solve('examples/afiro.mps')
        l.wait_for_job_end(id_not_inline)
        l.wait_for_job_end(id_inline)
        cls.id_not_inline = id_not_inline
        cls.id_inline = id_inline
        cls.lib = l

    @classmethod
    def tearDownClass(cls) -> None:
        cls.lib.delete_job(cls.id_inline, hard=True)
        cls.lib.delete_job(cls.id_not_inline, hard=True)

    def test_inline_details_dont_have_inputs_or_outputs_by_default(self):
        l = self.lib
        details = l.get_job_details(self.id_inline)
        self.assertNotIn('input_data_references',  details['entity']['decision_optimization'])
        self.assertNotIn('input_data',             details['entity']['decision_optimization'])
        self.assertNotIn('output_data_references', details['entity']['decision_optimization'])
        self.assertNotIn('output_data',            details['entity']['decision_optimization'])

    def test_non_inline_details_dont_have_inputs_or_outputs_by_default(self):
        l = self.lib
        details = l.get_job_details(self.id_not_inline)
        self.assertNotIn('input_data_references',  details['entity']['decision_optimization'])
        self.assertNotIn('input_data',             details['entity']['decision_optimization'])
        self.assertNotIn('output_data_references', details['entity']['decision_optimization'])
        self.assertNotIn('output_data',            details['entity']['decision_optimization'])

    def test_inline_details_do_have_inputs_and_outputs_if_names(self):
        l = self.lib
        id_inline = self.id_inline
        details = l.get_job_details(id_inline, with_contents='names')
        self.assertNotIn('input_data_references',  details['entity']['decision_optimization'])
        self.assertIn   ('input_data',             details['entity']['decision_optimization'])
        self.assertIn   ('output_data_references', details['entity']['decision_optimization'])
        # There is a list of output references, but it's empty
        self.assertEqual(len(details['entity']['decision_optimization']['output_data_references']), 0)
        self.assertIn   ('output_data',            details['entity']['decision_optimization'], )
        self.assertLogIsMentionedButNoContent(details)
        self.assertSolutionIsMentionedButNoContent(details)

    def test_non_inline_details_do_have_inputs_and_outputs_if_names(self):
        l = self.lib
        id_not_inline = self.id_not_inline
        details = l.get_job_details(id_not_inline, with_contents='names')
        self.assertIn   ('input_data_references',  details['entity']['decision_optimization'])
        self.assertNotIn('input_data',             details['entity']['decision_optimization'])
        self.assertIn   ('output_data_references', details['entity']['decision_optimization'])
        # There is a list of output references, but it's empty
        self.assertEqual(len(details['entity']['decision_optimization']['output_data_references']), 0)
        self.assertIn   ('output_data',            details['entity']['decision_optimization'], )
        self.assertLogIsMentionedButNoContent(details)
        self.assertSolutionIsMentionedButNoContent(details)

    # FIXME: check full details
    # FIXME: check presence or absence of log
    # FIXME: check content of input data only if inline
    # FIXME: check engine activity only in full


if __name__ == '__main__':
    main()
