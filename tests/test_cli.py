import os
import pprint
import sys
from collections import namedtuple
from logging import Logger
from unittest import TestCase, main, mock
from unittest.mock import Mock, ANY, call

from dowml import VERSION
from dowml.interactive import DOWMLInteractive, \
    CommandNeedsJobID, CommandNeedsNonNullInteger, InvalidArgumentForCommand
from dowml.dowmllib import DOWMLLib, version_is_greater

TEST_CREDENTIALS_FILE_NAME = 'test_credentials.txt'
EXPECTED = 'expected'


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class TestNumberToId(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive(TEST_CREDENTIALS_FILE_NAME)

    def test_return_last_job_id(self):
        with self.assertRaises(CommandNeedsJobID):
            self.cli._number_to_id(None)

    def test_there_must_be_a_previous(self):
        self.cli.last_job_id = EXPECTED
        result = self.cli._number_to_id(None)
        self.assertEqual(result, EXPECTED)

    def test_real_ids(self):
        self.cli.jobs = ['a', 'b']
        self.assertEqual(self.cli._number_to_id('a'), 'a')
        self.assertEqual(self.cli._number_to_id('b'), 'b')
        # Fall-through
        self.assertEqual(self.cli._number_to_id('c'), 'c')

    def test_numbers(self):
        self.cli.jobs = ['a', 'b']
        self.assertEqual(self.cli._number_to_id('1'), 'a')
        self.assertEqual(self.cli._number_to_id('2'), 'b')
        # out of list
        self.assertEqual(self.cli._number_to_id('3'), '3')


class TestDetails(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive(TEST_CREDENTIALS_FILE_NAME)
        self.cli.lib = Mock(spec=DOWMLLib)
        self.cli.jobs = ['a']
        self.cli.last_job_id = 'a'

    def test_with_no_content(self):
        mock_print = Mock(spec=pprint.pprint)
        expected = {'id': 'the_id'}
        self.cli.lib.get_job_details.return_value = expected
        self.cli.do_details('1', printer=mock_print)
        mock_print.assert_called_once_with(expected, indent=ANY, width=ANY)
        self.cli.lib.get_job_details.assert_called_once_with('a', with_contents=False)

    def test_with_names_only(self):
        mock_print = Mock(spec=pprint.pprint)
        expected = {'id': 'the_id'}
        self.cli.lib.get_job_details.return_value = expected
        self.cli.do_details('1 names', printer=mock_print)
        mock_print.assert_called_once_with(expected, indent=ANY, width=ANY)
        self.cli.lib.get_job_details.assert_called_once_with('a', with_contents='names')

    def with_content_helper(self, the_input):
        # It's not very nice to explicitly call setUp, but I'd rather not
        # create one test for each
        self.setUp()
        self.cli.do_details(the_input, printer=Mock(spec=pprint.pprint))
        self.cli.lib.get_job_details.assert_called_once_with('a', with_contents='full')

    def test_with_content_removed(self):
        self.with_content_helper('1 full')
        self.with_content_helper('full')
        self.with_content_helper('full 1')
        self.with_content_helper('full a')
        self.with_content_helper('a full')


class TestDelete(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive(TEST_CREDENTIALS_FILE_NAME)
        self.cli.lib = Mock(spec=DOWMLLib)
        JobTuple = namedtuple('Job', ['id'])
        self.cli.lib.get_jobs = Mock(return_value=[JobTuple(id='a'), JobTuple(id='b'), JobTuple(id='c')])
        self.cli.jobs = ['a', 'b', 'c']

    def test_delete_first(self):
        self.cli.do_delete('1')
        self.cli.lib.delete_job.assert_called_once_with('a', True)
        self.assertEqual(self.cli.jobs, ['b', 'c'])

    def test_delete_not_in_list(self):
        self.cli.do_delete('d')
        self.cli.lib.delete_job.assert_called_once_with('d', True)
        self.assertEqual(self.cli.jobs, ['a', 'b', 'c'])

    def test_delete_all(self):
        self.cli.do_delete('*')
        self.cli.lib.get_jobs.assert_called_once()
        self.cli.lib.delete_job.assert_has_calls([
            call('a', True),
            call('b', True),
            call('c', True)
        ], any_order=True)
        self.assertEqual(self.cli.jobs, [])

    def test_delete_current(self):
        self.cli.last_job_id = 'b'
        self.cli.do_delete('')
        self.cli.lib.delete_job.assert_called_once_with('b', True)
        self.assertEqual(1, self.cli.lib.delete_job.call_count)
        self.assertEqual(self.cli.jobs, ['a', 'c'])
        self.assertIsNone(self.cli.last_job_id)

    def test_delete_not_current(self):
        self.cli.last_job_id = 'b'
        self.cli.do_delete('c')
        self.assertEqual(self.cli.last_job_id, 'b')

    def test_delete_range(self):
        self.cli.do_delete('1-2')
        self.cli.lib.delete_job.assert_has_calls([
            call('a', True),
            call('b', True)
        ], any_order=True)
        self.assertEqual(2, self.cli.lib.delete_job.call_count)

    def test_delete_range_complete(self):
        self.cli.do_delete('1-3')
        self.cli.lib.delete_job.assert_has_calls([
            call('a', True),
            call('b', True),
            call('c', True)
        ], any_order=True)
        self.assertEqual(3, self.cli.lib.delete_job.call_count)

    def test_delete_range_empty(self):
        self.cli.do_delete('2-1')
        self.cli.lib.delete_job.assert_not_called()

    def test_delete_range_larger(self):
        self.cli.do_delete('3-5')
        self.cli.lib.delete_job.assert_has_calls([
            call('c', True)
        ], any_order=True)
        self.assertEqual(1, self.cli.lib.delete_job.call_count)


class TestOutput(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive(TEST_CREDENTIALS_FILE_NAME)
        self.cli.lib = Mock(spec=DOWMLLib)
        self.cli.lib.get_job_details.return_value = {}
        self.cli.jobs = ['a', 'b', 'c']

    def test_command_exists(self):
        self.cli.lib.get_output.return_value = []
        mock_open = mock.mock_open()
        mock_mkdir = Mock()
        with mock.patch('os.mkdir', mock_mkdir):
            with mock.patch('builtins.open', mock_open):
                self.cli.do_output('1')

    def test_output_asks_for_tabular_outputs_as_csv(self):
        mock_get_output = Mock()
        mock_get_output.return_value = {}
        self.cli.lib.get_output = mock_get_output
        mock_open = mock.mock_open()
        mock_mkdir = Mock()
        with mock.patch('os.mkdir', mock_mkdir):
            with mock.patch('builtins.open', mock_open):
                self.cli.do_output('1')
        mock_get_output.assert_called_once_with({}, tabular_as_csv=True)

    def test_store_files(self):
        self.cli.lib.get_output.return_value = {
            'out1': b'content-a',
            'out2': b'content-b'
        }
        self.cli.save_content = Mock()
        mock_open = mock.mock_open()
        mock_mkdir = Mock()
        with mock.patch('os.mkdir', mock_mkdir):
            with mock.patch('builtins.open', mock_open):
                self.cli.do_output('1')
        self.cli.save_content.assert_has_calls([
            call('a', 'out1', b'content-a'),
            call('a', 'out2', b'content-b'),
            call('a', 'details.json', '{}', text=True)
        ], any_order=True)

    def test_save_content(self):
        write_data = b'content'
        mock_open = mock.mock_open()
        mock_mkdir = Mock()
        # It is necessary to create the HiddenPrint first, because that action
        # calls the built-in 'open'. And if that one is mocked first, not only
        # will the HiddenPrint not really work as expected, but the test will
        # fail as mock_open will have been called twice instead of once.
        with HiddenPrints(), \
                mock.patch('os.mkdir', mock_mkdir), \
                mock.patch('builtins.open', mock_open):
            self.cli.save_content('id', 'name', write_data)
        mock_mkdir.assert_called_once_with('id')
        mock_open.assert_called_once_with('id/name', 'wb')
        # noinspection PyArgumentList
        handle = mock_open()
        # noinspection PyUnresolvedReferences
        handle.write.assert_called_once_with(write_data)

    def test_save_two_files_under_same_id_doesnt_throw(self):
        write_data = b'content'
        mock_open = mock.mock_open()
        mock_mkdir = Mock()
        with HiddenPrints(), \
                mock.patch('os.mkdir', mock_mkdir), \
                mock.patch('builtins.open', mock_open):
            self.cli.lib._logger = Mock(spec=Logger)
            self.cli.save_content('id', 'name1', write_data)
            self.cli.save_content('id', 'name2', write_data)
        mock_mkdir.assert_has_calls([call('id'), call('id')])


class TestOneJobShouldBeCurrent(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive(TEST_CREDENTIALS_FILE_NAME)
        self.cli.lib = Mock(spec=DOWMLLib)
        self.cli.jobs = ['a', 'b']

    def test_delete_leaves_last(self):
        cli = self.cli
        cli.last_job_id = None
        cli.do_delete('1')
        self.assertEqual(cli.last_job_id, 'b')

    def test_delete_leaves_first(self):
        cli = self.cli
        cli.last_job_id = None
        cli.do_delete('2')
        self.assertEqual(cli.last_job_id, 'a')

    def test_only_one_job(self):
        cli = self.cli
        JobTuple = namedtuple('Job', ['status', 'id', 'created', 'names', 'type', 'version', 'size'])
        jobs = [
            JobTuple(status='', id='a', created='', names='', type='', version='', size=''),
        ]
        cli.lib.get_jobs.return_value = jobs
        assert cli.last_job_id is None
        cli.do_jobs('')
        self.assertEqual(cli.last_job_id, 'a')


class TestTimeLimit(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive(TEST_CREDENTIALS_FILE_NAME)

    def test_default_is_no_timelimit(self):
        cli = self.cli
        self.assertIsNone(cli.lib.timelimit)

    def test_setting_timelimit(self):
        cli = self.cli
        cli.do_time('10')
        self.assertEqual(cli.lib.timelimit, 10)

    def test_cant_set_to_non_integer(self):
        cli = self.cli
        with self.assertRaises(CommandNeedsNonNullInteger):
            cli.do_time('foo')

    def test_cant_set_to_negative_integer(self):
        cli = self.cli
        with self.assertRaises(CommandNeedsNonNullInteger):
            cli.do_time('-1')

    def test_zero_means_no_timelimit(self):
        cli = self.cli
        cli.do_time('0')
        self.assertIsNone(cli.lib.timelimit)


class TestInline(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive(TEST_CREDENTIALS_FILE_NAME)

    def test_remove_after_deprecation(self):
        self.assertFalse(version_is_greater(VERSION, "2.0"),
                         "'inline' was deprecated long enough and should be removed")

    def test_default_is_not_inline(self):
        cli = self.cli
        self.cli.lib._logger = Mock(spec=Logger)
        self.assertFalse(cli.lib.inline)
        self.cli.lib._logger.warning.assert_called_once()

    def test_yes_or_no_only(self):
        cli = self.cli
        with self.assertRaises(InvalidArgumentForCommand):
            cli.do_inline('y')


class TestInputs(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive(TEST_CREDENTIALS_FILE_NAME)

    def test_default_is_not_inline(self):
        cli = self.cli
        self.assertEqual('assets', cli.lib.inputs)

    def test_can_set_inputs_to_assets_or_inline(self):
        cli = self.cli
        cli.do_inputs('inline')
        self.assertEqual('inline', cli.lib.inputs)
        cli.do_inputs('assets')
        self.assertEqual('assets', cli.lib.inputs)

    def test_cant_set_inputs_to_arbitrary_value(self):
        cli = self.cli
        with self.assertRaises(InvalidArgumentForCommand):
            cli.do_inputs('foo')


class TestOutputs(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive(TEST_CREDENTIALS_FILE_NAME)

    def test_default_outputs_is_inline(self):
        cli = self.cli
        self.assertEqual('inline', cli.lib.outputs)

    def test_can_set_outputs_to_assets_or_inline(self):
        cli = self.cli
        cli.do_outputs('assets')
        self.assertEqual('assets', cli.lib.outputs)
        cli.do_outputs('inline')
        self.assertEqual('inline', cli.lib.outputs)

    def test_cant_set_outputs_to_arbitrary_value(self):
        cli = self.cli
        with self.assertRaises(InvalidArgumentForCommand):
            cli.do_outputs('foo')


class TestShellCommand(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive(TEST_CREDENTIALS_FILE_NAME)

    def test_successful_shell_returns_zero(self):
        self.assertEqual(0, self.cli.do_shell('ls > /dev/null'))

    def test_invalid_command_returns_zero(self):
        # May seem counter-intuitive, but anything else would
        # stop the Interactive
        self.assertEqual(0, self.cli.do_shell('foo 2>/dev/null'))

    def test_empty_command_returns_zero(self):
        self.assertEqual(0, self.cli.do_shell(''))


if __name__ == '__main__':
    main()
