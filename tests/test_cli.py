import pprint
from unittest import TestCase, main, mock
from unittest.mock import Mock, ANY, call

from dowml import DOWMLInteractive, CommandNeedsJobID
from dowmllib import DOWMLLib

EXPECTED = 'expected'


class TestNumberToId(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive('test_credentials.txt')

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
        self.cli = DOWMLInteractive('test_credentials.txt')
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

    def with_content_helper(self, the_input):
        # It's not very nice to explicitly call setUp, but I'd rather not
        # create one test for each
        self.setUp()
        self.cli.do_details(the_input, printer=Mock(spec=pprint.pprint))
        self.cli.lib.get_job_details.assert_called_once_with('a', with_contents=True)

    def test_with_content_removed(self):
        self.with_content_helper('1 full')
        self.with_content_helper('full')
        self.with_content_helper('full 1')
        self.with_content_helper('full a')
        self.with_content_helper('a full')


class TestJobs(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive('test_credentials.txt')
        self.cli.lib = Mock(spec=DOWMLLib)
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
        self.assertEqual(self.cli.jobs, ['a', 'c'])
        self.assertIsNone(self.cli.last_job_id)

    def test_delete_not_current(self):
        self.cli.last_job_id = 'b'
        self.cli.do_delete('c')
        self.assertEqual(self.cli.last_job_id, 'b')


class TestOutput(TestCase):

    def setUp(self) -> None:
        self.cli = DOWMLInteractive('test_credentials.txt')
        self.cli.lib = Mock(spec=DOWMLLib)
        self.cli.jobs = ['a', 'b', 'c']

    def test_command_exists(self):
        self.cli.lib.get_output.return_value = []
        self.cli.do_output('1')

    def test_store_files(self):
        self.cli.lib.get_output.return_value = [
            ('out1', b'content-a'),
            ('out2', b'content-b'),
        ]
        self.cli.save_content = Mock()
        self.cli.do_output('1')
        self.cli.save_content.assert_has_calls([
            call('a', 'out1', b'content-a'),
            call('a', 'out2', b'content-b')
        ])

    def test_save_content(self):
        write_data = b'content'
        mock_open = mock.mock_open()
        with mock.patch('builtins.open', mock_open) as m:
            self.cli.save_content('id', 'name', write_data)
        m.assert_called_once_with('id_name', 'wb')
        # noinspection PyArgumentList
        handle = m()
        # noinspection PyUnresolvedReferences
        handle.write.assert_called_once_with(write_data)


if __name__ == '__main__':
    main()
