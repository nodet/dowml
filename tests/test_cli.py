import pprint
from unittest import TestCase, main
from unittest.mock import Mock, ANY, call

from cli import DOWMLInteractive, CommandNeedsJobID
from dowmlclient import DOWMLClient

EXPECTED = 'expected'


class TestNumberToId(TestCase):

    def setUp(self) -> None:
        self.client = DOWMLInteractive('test_credentials.txt')

    def test_return_last_job_id(self):
        with self.assertRaises(CommandNeedsJobID):
            self.client._number_to_id(None)

    def test_there_must_be_a_previous(self):
        self.client.last_job_id = EXPECTED
        result = self.client._number_to_id(None)
        self.assertEqual(result, EXPECTED)

    def test_real_ids(self):
        self.client.jobs = ['a', 'b']
        self.assertEqual(self.client._number_to_id('a'), 'a')
        self.assertEqual(self.client._number_to_id('b'), 'b')
        # Fall-through
        self.assertEqual(self.client._number_to_id('c'), 'c')

    def test_numbers(self):
        self.client.jobs = ['a', 'b']
        self.assertEqual(self.client._number_to_id('1'), 'a')
        self.assertEqual(self.client._number_to_id('2'), 'b')
        # out of list
        self.assertEqual(self.client._number_to_id('3'), '3')


class TestDetails(TestCase):

    def setUp(self) -> None:
        self.client = DOWMLInteractive('test_credentials.txt')
        self.client.client = Mock(spec=DOWMLClient)
        self.client.jobs = ['a']
        self.client.last_job_id = 'a'

    def test_with_no_content(self):
        mock_print = Mock(spec=pprint.pprint)
        expected = {'id': 'the_id'}
        self.client.client.get_job_details.return_value = expected
        self.client.do_details('1', printer=mock_print)
        mock_print.assert_called_once_with(expected, indent=ANY, width=ANY)
        self.client.client.get_job_details.assert_called_once_with('a', with_contents=False)

    def with_content_helper(self, input):
        # It's not very nice to explicitly call setUp, but I'd rather not
        # create one test for each
        self.setUp()
        self.client.do_details(input, printer=Mock(spec=pprint.pprint))
        self.client.client.get_job_details.assert_called_once_with('a', with_contents=True)

    def test_with_content_removed(self):
        self.with_content_helper('1 full')
        self.with_content_helper('full')
        self.with_content_helper('full 1')
        self.with_content_helper('full a')
        self.with_content_helper('a full')


class TestJobs(TestCase):

    def setUp(self) -> None:
        self.client = DOWMLInteractive('test_credentials.txt')
        self.client.client = Mock(spec=DOWMLClient)
        self.client.jobs = ['a', 'b', 'c']

    def test_delete_first(self):
        self.client.do_delete('1')
        self.client.client.delete_job.assert_called_once_with('a', True)
        self.assertEqual(self.client.jobs, ['b', 'c'])

    def test_delete_not_in_list(self):
        self.client.do_delete('d')
        self.client.client.delete_job.assert_called_once_with('d', True)
        self.assertEqual(self.client.jobs, ['a', 'b', 'c'])

    def test_delete_all(self):
        self.client.do_delete('*')
        self.client.client.delete_job.assert_has_calls([
            call('a', True),
            call('b', True),
            call('c', True)
        ], any_order=True)
        self.assertEqual(self.client.jobs, [])

    def test_delete_current(self):
        self.client.last_job_id = 'b'
        self.client.do_delete('')
        self.client.client.delete_job.assert_called_once_with('b', True)
        self.assertEqual(self.client.jobs, ['a', 'c'])
        self.assertIsNone(self.client.last_job_id)

    def test_delete_not_current(self):
        self.client.last_job_id = 'b'
        self.client.do_delete('c')
        self.assertEqual(self.client.last_job_id, 'b')


if __name__ == '__main__':
    main()
