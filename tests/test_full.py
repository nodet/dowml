import pprint

from ibm_watson_machine_learning.wml_client_error import ApiRequestFailure
from pandas import DataFrame

from dowml.dowmllib import DOWMLLib
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
        self.job_solves_to_completion(inline=True, paths='../examples/afiro.mps')

    def test_docplex_inline(self):
        # 'type docplex' 'solve examples/markshare.py examples/markshare1.mps.gz' wait jobs output 'details full' delete
        self.job_solves_to_completion(inline=True,
                                      type='docplex',
                                      paths='../examples/markshare.py ../examples/markshare1.mps.gz')


class TestDetailsAndOutputs(TestCase):
    """"This class runs two CPLEX models, one inline, the other by reference.
    And it checks that the details and outputs are as expected"""

    def assertOutputIsMentionedButNoContent(self, details, name):
        output = details['entity']['decision_optimization']['output_data']
        seen_output = False
        for o in output:
            if o['id'] == name:
                self.assertFalse(seen_output)
                seen_output = True
                content = o['content']
                self.assertEqual(content, '[not shown]')

    def assertLogIsMentionedButNoContent(self, details):
        self.assertOutputIsMentionedButNoContent(details, 'log.txt')

    def assertSolutionIsMentionedButNoContent(self, details):
        self.assertOutputIsMentionedButNoContent(details, 'solution.json')

    def assertOutputHasContent(self, details, name):
        output = details['entity']['decision_optimization']['output_data']
        seen_output = False
        for o in output:
            if o['id'] == name:
                self.assertFalse(seen_output)
                seen_output = True
                content = o['content']
                self.assertNotEqual(content, '[not shown]')

    def assertLogHasContent(self, details):
        self.assertOutputHasContent(details, 'log.txt')

    def assertSolutionHasContent(self, details):
        self.assertOutputHasContent(details, 'solution.json')

    def assertEngineActivityButNoContent(self, details):
        self.assertIn('solve_state', details['entity']['decision_optimization'])
        self.assertIn('latest_engine_activity', details['entity']['decision_optimization']['solve_state'])
        activity = details['entity']['decision_optimization']['solve_state']['latest_engine_activity']
        self.assertEqual(activity, ['[not shown]'])

    def assertStatsAreMentionedButNoContent(self, details):
        output = details['entity']['decision_optimization']['output_data']
        seen_output = False
        for o in output:
            if o['id'] == 'stats.csv':
                self.assertFalse(seen_output)
                seen_output = True
                values = o['values']
                self.assertEqual(values, ['[not shown]'])

    def assertEngineActivityWithContent(self, details):
        self.assertIn('solve_state', details['entity']['decision_optimization'])
        self.assertIn('latest_engine_activity', details['entity']['decision_optimization']['solve_state'])
        activity = details['entity']['decision_optimization']['solve_state']['latest_engine_activity']
        self.assertNotEqual(activity, ['[not shown]'])

    def assertStatsAreMentionedWithContent(self, details):
        output = details['entity']['decision_optimization']['output_data']
        seen_output = False
        for o in output:
            if o['id'] == 'stats.csv':
                self.assertFalse(seen_output)
                seen_output = True
                values = o['values']
                self.assertNotEqual(values, ['[not shown]'])

    def assertNoInputDataReference(self, details):
        self.assertNotIn('input_data_references',  details['entity']['decision_optimization'])

    def assertInputDataReference(self, details):
        self.assertIn('input_data_references',  details['entity']['decision_optimization'])

    def assertNoInputData(self, details):
        self.assertNotIn('input_data',  details['entity']['decision_optimization'])

    def assertInputData(self, details):
        self.assertIn('input_data',  details['entity']['decision_optimization'])

    def assertOutputDataReference(self, details):
        self.assertIn('output_data_references',  details['entity']['decision_optimization'])

    def assertNoOutputDataReference(self, details):
        self.assertNotIn('output_data_references',  details['entity']['decision_optimization'])

    def assertEmptyOutputDataReference(self, details):
        self.assertOutputDataReference(details)
        self.assertEqual(len(details['entity']['decision_optimization']['output_data_references']), 0)

    def assertNoOutputData(self, details):
        self.assertNotIn('output_data',  details['entity']['decision_optimization'])

    @classmethod
    def setUpClass(cls) -> None:
        l = DOWMLLib()
        id_not_inline = l.solve('../examples/afiro.mps')
        l.inline = True
        id_inline = l.solve('../examples/afiro.mps')
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
        self.assertNoInputDataReference(details)
        self.assertNoInputData(details)
        self.assertNoOutputDataReference(details)
        self.assertNoOutputData(details)

    def test_non_inline_details_dont_have_inputs_or_outputs_by_default(self):
        l = self.lib
        details = l.get_job_details(self.id_not_inline)
        self.assertNoInputDataReference(details)
        self.assertNoInputData(details)
        self.assertNoOutputDataReference(details)
        self.assertNoOutputData(details)

    def test_inline_details_do_have_inputs_and_outputs_if_names(self):
        l = self.lib
        id_inline = self.id_inline
        details = l.get_job_details(id_inline, with_contents='names')
        self.assertNoInputDataReference(details)
        self.assertInputData(details)
        self.assertEqual(details['entity']['decision_optimization']['input_data'],
                         [{'content': '[not shown]', 'id': 'afiro.mps'}])
        self.assertEmptyOutputDataReference(details)
        self.assertLogIsMentionedButNoContent(details)
        self.assertSolutionIsMentionedButNoContent(details)
        self.assertStatsAreMentionedButNoContent(details)
        self.assertEngineActivityButNoContent(details)

    def test_non_inline_details_do_have_inputs_and_outputs_if_names(self):
        l = self.lib
        id_not_inline = self.id_not_inline
        details = l.get_job_details(id_not_inline, with_contents='names')
        self.assertInputDataReference(details)
        self.assertNoInputData(details)
        self.assertEmptyOutputDataReference(details)
        self.assertLogIsMentionedButNoContent(details)
        self.assertSolutionIsMentionedButNoContent(details)
        self.assertStatsAreMentionedButNoContent(details)
        self.assertEngineActivityButNoContent(details)

    def assert_full_details_have_all_info(self, details):
        self.assertLogHasContent(details)
        self.assertSolutionHasContent(details)
        self.assertStatsAreMentionedWithContent(details)
        self.assertEngineActivityWithContent(details)

    def test_inline_details_full(self):
        l = self.lib
        id_inline = self.id_inline
        details = l.get_job_details(id_inline, with_contents='full')
        self.assertNoInputDataReference(details)
        self.assertInputData(details)
        input_list = details['entity']['decision_optimization']['input_data']
        self.assertEqual(len(input_list), 1)
        input = input_list[0]
        self.assertIn('id', input)
        self.assertEqual('afiro.mps', input['id'])
        self.assertIn('content', input)
        self.assertNotEqual('[not shown]', input['content'])
        self.assertEmptyOutputDataReference(details)
        self.assert_full_details_have_all_info(details)

    def test_non_inline_details_full(self):
        l = self.lib
        id_not_inline = self.id_not_inline
        details = l.get_job_details(id_not_inline, with_contents='full')
        self.assertInputDataReference(details)
        self.assertNoInputData(details)
        self.assertEmptyOutputDataReference(details)
        self.assert_full_details_have_all_info(details)

    def find_output_with_id(self, outputs, id):
        result = None
        for name in outputs:
            if name == id:
                # We should find at most one
                self.assertIsNone(result)
                result = outputs[name]
        # We should find at least one
        self.assertIsNotNone(result)
        return result

    def test_log_is_correctly_decoded(self):
        l = self.lib
        id_not_inline = self.id_not_inline
        details = l.get_job_details(id_not_inline, with_contents='full')
        outputs = l.get_output(details)
        log = self.find_output_with_id(outputs, 'log.txt')
        self.assertEqual(b'CPLEX version', log[29:42])

    def test_csv_is_correctly_decoded(self):
        l = self.lib
        id_not_inline = self.id_not_inline
        details = l.get_job_details(id_not_inline, with_contents='full')
        outputs = l.get_output(details, tabular_as_csv=True)
        csv = self.find_output_with_id(outputs, 'stats.csv')
        self.assertEqual(b'Name,Value\r\n', csv[0:12])

    def test_csv_is_correctly_decoded_as_dataframe(self):
        l = self.lib
        id_not_inline = self.id_not_inline
        details = l.get_job_details(id_not_inline, with_contents='full')
        outputs = l.get_output(details)
        csv = self.find_output_with_id(outputs, 'stats.csv')
        self.assertEqual(DataFrame, type(csv))
        self.assertEqual(['Name', 'Value'], list(csv.columns))
        # There exists at least one line with 'job.coresCount' in the 'Name'
        # column, and that line has '1' in the 'Value' column
        self.assertEqual(1, csv.loc[csv['Name'] == 'job.coresCount']['Value'].values[0])


if __name__ == '__main__':
    main()
