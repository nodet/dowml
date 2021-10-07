import os
import shutil

from pandas import DataFrame

from dowml.dowmllib import DOWMLLib
from unittest import TestCase, main

from dowml.interactive import DOWMLInteractive


class TestDetailsAndOutputs(TestCase):
    """"This class runs two CPLEX models, one inline, the other by reference.
    And it checks that the details and outputs are as expected"""

    lib = None
    id_not_inline = None
    id_inline = None

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

    def assertNonEmptyOutputDataReference(self, details):
        self.assertOutputDataReference(details)
        self.assertNotEqual(len(details['entity']['decision_optimization']['output_data_references']), 0)

    def assertNoOutputData(self, details):
        self.assertNotIn('output_data', details['entity']['decision_optimization'])

    def assertHasSolveState(self, details):
        self.assertIn('solve_state', details['entity']['decision_optimization'])

    def assertHasStatus(self, details):
        self.assertIn('status', details['entity']['decision_optimization'])

    @classmethod
    def setUpClass(cls) -> None:
        lib = DOWMLLib()
        lib.outputs = 'assets'
        id_not_inline = lib.solve('../examples/afiro.mps')
        lib.inputs = 'inline'
        lib.outputs = 'inline'
        id_inline = lib.solve('../examples/afiro.mps')
        lib.wait_for_job_end(id_not_inline)
        lib.wait_for_job_end(id_inline)
        cls.id_not_inline = id_not_inline
        cls.id_inline = id_inline
        cls.lib = lib

    @classmethod
    def tearDownClass(cls) -> None:
        cls.lib.delete_job(cls.id_inline, hard=True)
        cls.lib.delete_job(cls.id_not_inline, hard=True)

        def remove_if_exists(name):
            try:
                shutil.rmtree(name)
            except FileNotFoundError:
                pass

        remove_if_exists(cls.id_inline)
        remove_if_exists(cls.id_not_inline)

    def test_default_details_have_solve_state(self):
        lib = self.lib
        details = lib.get_job_details(self.id_inline)
        self.assertHasSolveState(details)

    def test_default_details_have_status(self):
        lib = self.lib
        details = lib.get_job_details(self.id_inline)
        self.assertHasStatus(details)

    def test_full_details_have_solve_state(self):
        lib = self.lib
        details = lib.get_job_details(self.id_inline, with_contents='full')
        self.assertHasSolveState(details)

    def test_full_details_have_status(self):
        lib = self.lib
        details = lib.get_job_details(self.id_inline, with_contents='full')
        self.assertHasStatus(details)

    def test_inline_details_dont_have_inputs_or_outputs_by_default(self):
        lib = self.lib
        details = lib.get_job_details(self.id_inline)
        self.assertNoInputDataReference(details)
        self.assertNoInputData(details)
        self.assertNoOutputDataReference(details)
        self.assertNoOutputData(details)

    def test_non_inline_details_dont_have_inputs_or_outputs_by_default(self):
        lib = self.lib
        details = lib.get_job_details(self.id_not_inline)
        self.assertNoInputDataReference(details)
        self.assertNoInputData(details)
        self.assertNoOutputDataReference(details)
        self.assertNoOutputData(details)

    def test_inline_details_do_have_inputs_and_outputs_if_names(self):
        lib = self.lib
        id_inline = self.id_inline
        details = lib.get_job_details(id_inline, with_contents='names')
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
        lib = self.lib
        id_not_inline = self.id_not_inline
        details = lib.get_job_details(id_not_inline, with_contents='names')
        self.assertInputDataReference(details)
        self.assertNoInputData(details)
        self.assertNonEmptyOutputDataReference(details)
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
        lib = self.lib
        id_inline = self.id_inline
        details = lib.get_job_details(id_inline, with_contents='full')
        self.assertNoInputDataReference(details)
        self.assertInputData(details)
        input_list = details['entity']['decision_optimization']['input_data']
        self.assertEqual(len(input_list), 1)
        the_input = input_list[0]
        self.assertIn('id', the_input)
        self.assertEqual('afiro.mps', the_input['id'])
        self.assertIn('content', the_input)
        self.assertNotEqual('[not shown]', the_input['content'])
        self.assertEmptyOutputDataReference(details)
        self.assert_full_details_have_all_info(details)

    def test_non_inline_details_full(self):
        lib = self.lib
        id_not_inline = self.id_not_inline
        details = lib.get_job_details(id_not_inline, with_contents='full')
        self.assertInputDataReference(details)
        self.assertNoInputData(details)
        self.assertNonEmptyOutputDataReference(details)
        self.assert_full_details_have_all_info(details)

    def find_output_with_id(self, outputs, job_id):
        result = None
        for name in outputs:
            if name == job_id:
                # We should find at most one
                self.assertIsNone(result)
                result = outputs[name]
        # We should find at least one
        self.assertIsNotNone(result)
        return result

    def test_inline_log_is_correctly_decoded(self):
        lib = self.lib
        details = lib.get_job_details(self.id_inline, with_contents='full')
        outputs = lib.get_output(details)
        log = self.find_output_with_id(outputs, 'log.txt')
        self.assertEqual(b'CPLEX version', log[29:42])

    def test_inline_csv_is_correctly_decoded(self):
        lib = self.lib
        details = lib.get_job_details(self.id_inline, with_contents='full')
        outputs = lib.get_output(details, tabular_as_csv=True)
        csv = self.find_output_with_id(outputs, 'stats.csv')
        self.assertEqual(b'Name,Value\r\n', csv[0:12])

    def test_inline_csv_is_correctly_decoded_as_dataframe(self):
        lib = self.lib
        details = lib.get_job_details(self.id_inline, with_contents='full')
        outputs = lib.get_output(details)
        csv = self.find_output_with_id(outputs, 'stats.csv')
        self.assertEqual(DataFrame, type(csv))
        self.assertEqual(['Name', 'Value'], list(csv.columns))
        # There exists at least one line with 'job.coresCount' in the 'Name'
        # column, and that line has '1' in the 'Value' column
        self.assertEqual(1, csv.loc[csv['Name'] == 'job.coresCount']['Value'].values[0])

    def check_stored_files(self, cli, job_id, files):
        cli.do_output(job_id)
        for f in files:
            self.assertTrue(os.path.isfile(f'{job_id}/{f}'))
        self.assertEqual(len(files), len(os.listdir(job_id)))

    def test_output_for_inline_has_expected_files(self):
        cli = DOWMLInteractive(wml_cred_file=None)
        self.check_stored_files(cli, self.id_inline, ['solution.json', 'stats.csv', 'log.txt', 'details.json'])


if __name__ == '__main__':
    main()
