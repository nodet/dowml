from unittest import TestCase, main

from cli import DOWMLInteractive, CommandNeedsJobID

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


if __name__ == '__main__':
    main()