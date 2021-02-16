from logging import Logger
from unittest.mock import Mock

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.deployments import Deployments

from dowmllib import DOWMLLib, SimilarNamesInJob
from unittest import TestCase, main


class TestSolve(TestCase):

    def setUp(self) -> None:
        client = DOWMLLib('test_credentials.txt')
        client._logger = Mock(spec=Logger)
        client._client = Mock(spec=APIClient)
        client._client.deployments = Mock(spec=Deployments)
        domn = Mock()
        domn.SOLVE_PARAMETERS = 'solve-parameters'
        domn.INPUT_DATA = 'input-data'
        domn.OUTPUT_DATA = 'output-data'
        client._client.deployments.DecisionOptimizationMetaNames = domn
        client.get_file_as_data = lambda path: 'base-64-content'
        client._space_id = 'space-id'
        client._get_deployment_id = Mock(spec=DOWMLLib._get_deployment_id)
        client._get_deployment_id.return_value = 'deployment-id'
        self.client = client

    def test_solve_single_file(self):
        self.client.solve('afiro.mps')
        create_job_mock = self.client._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(kall.kwargs, {})
        self.assertEqual(kall.args[0], 'deployment-id')
        self.assertEqual(len(kall.args[1]['input-data']), 1)
        i = kall.args[1]['input-data'][0]
        self.assertEqual(i['content'], 'base-64-content')
        self.assertEqual(i['id'], 'afiro.mps')

    def test_solve_multiple_files(self):
        self.client.solve('f1.lp f2.prm')
        create_job_mock = self.client._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(kall.args[1]['input-data'][0]['id'], 'f1.lp')
        self.assertEqual(kall.args[1]['input-data'][1]['id'], 'f2.prm')

    def test_solve_with_path(self):
        self.client.solve('/this/is/an/absolute/path')
        create_job_mock = self.client._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(kall.args[1]['input-data'][0]['id'], 'path')

    def test_solve_relative_path(self):
        self.client.solve('this/is/a/relative/path')
        create_job_mock = self.client._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(kall.args[1]['input-data'][0]['id'], 'path')

    def test_solve_same_names(self):
        with self.assertRaises(SimilarNamesInJob):
            self.client.solve('path/f1 another/path/f1')


if __name__ == '__main__':
    main()
