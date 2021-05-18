from logging import Logger
from unittest.mock import Mock

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.deployments import Deployments

from dowmllib import DOWMLLib, SimilarNamesInJob
from unittest import TestCase, main


class TestSolve(TestCase):

    def setUp(self) -> None:
        lib = DOWMLLib('test_credentials.txt')
        lib._logger = Mock(spec=Logger)
        lib._client = Mock(spec=APIClient)
        lib._client.deployments = Mock(spec=Deployments)
        lib.inline = True
        domn = Mock()
        domn.SOLVE_PARAMETERS = 'solve-parameters'
        domn.INPUT_DATA = 'input-data'
        domn.OUTPUT_DATA = 'output-data'
        lib._client.deployments.DecisionOptimizationMetaNames = domn
        lib.get_file_as_data = lambda path: 'base-64-content'
        lib._space_id = 'space-id'
        lib._get_deployment_id = Mock(spec=DOWMLLib._get_deployment_id)
        lib._get_deployment_id.return_value = 'deployment-id'
        self.lib = lib

    def test_solve_single_file(self):
        self.lib.solve('afiro.mps')
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(kall.kwargs, {})
        self.assertEqual(kall.args[0], 'deployment-id')
        self.assertEqual(len(kall.args[1]['input-data']), 1)
        i = kall.args[1]['input-data'][0]
        self.assertEqual(i['content'], 'base-64-content')
        self.assertEqual(i['id'], 'afiro.mps')

    def test_solve_multiple_files(self):
        self.lib.solve('f1.lp f2.prm')
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(kall.args[1]['input-data'][0]['id'], 'f1.lp')
        self.assertEqual(kall.args[1]['input-data'][1]['id'], 'f2.prm')

    def test_solve_with_path(self):
        self.lib.solve('/this/is/an/absolute/path')
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(kall.args[1]['input-data'][0]['id'], 'path')

    def test_solve_relative_path(self):
        self.lib.solve('this/is/a/relative/path')
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(kall.args[1]['input-data'][0]['id'], 'path')

    def test_solve_same_names(self):
        with self.assertRaises(SimilarNamesInJob):
            self.lib.solve('path/f1 another/path/f1')

    @staticmethod
    def get_params(create_job_mock):
        return create_job_mock.call_args.args[1]['solve-parameters']

    def test_solve_defaults_to_no_timelimit(self):
        self.assertIsNone(self.lib.timelimit)
        self.lib.solve('path')
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        self.assertNotIn('oaas.timeLimit', self.get_params(create_job_mock))

    def test_solve_defaults_with_time_limit(self):
        lim = 10
        self.lib.timelimit = lim
        self.lib.solve('path')
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        params = self.get_params(create_job_mock)
        self.assertIn('oaas.timeLimit', params)
        self.assertEqual(params['oaas.timeLimit'], 1000*lim)

    def test_solve_defaults_with_zero_limit(self):
        self.lib.timelimit = 0
        self.lib.solve('path')
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        self.assertNotIn('oaas.timeLimit', self.get_params(create_job_mock))


    def test_get_csv_file(self):
        details = {
            'entity': {
                'decision_optimization': {
                    'output_data': [
                       {
                          'fields': ['i', 'f'],
                          'id': 'results.csv',
                          'values': [
                             [0, 0],
                             ['a,b', 'c d']
                          ]
                       }
                    ]
                }
            }
        }
        output = self.lib.get_output(details)
        output_data = details['entity']['decision_optimization']['output_data']
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0][0], 'results.csv')
        lines = output[0][1].decode().split('\r\n')
        # There is one empty line at the end
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], 'i,f')
        self.assertEqual(lines[1], '0,0')
        self.assertEqual(lines[2], '"a,b",c d')

if __name__ == '__main__':
    main()
