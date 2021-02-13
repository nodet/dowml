import pprint
from unittest import TestCase, main
from unittest.mock import Mock, ANY

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
        self.jobs = ['a']
        self.client.last_job_id = 1

    def test_without_full(self):
        mock_print = Mock(spec=pprint.pprint)
        mock_client = Mock(spec=DOWMLClient)
        EXPECTED = {'id': 'the_id'}
        mock_client.get_job_details.return_value = EXPECTED
        self.client.client = mock_client
        self.client.do_details('1', printer=mock_print)
        mock_print.assert_called_once_with(EXPECTED, indent=ANY, width=ANY)


if __name__ == '__main__':
    main()