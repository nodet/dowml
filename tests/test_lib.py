import pprint
from logging import Logger
from unittest.mock import Mock

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.deployments import Deployments
from ibm_watson_machine_learning.assets import Assets

from dowmllib import DOWMLLib, SimilarNamesInJob
from unittest import TestCase, main


class TestSolveInline(TestCase):

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


class TestSolveUsingDataAssets(TestCase):

    def setUp(self) -> None:
        lib = DOWMLLib('test_credentials.txt')
        lib._logger = Mock(spec=Logger)
        lib._client = Mock(spec=APIClient)
        lib._client.deployments = Mock(spec=Deployments)
        # We assume having a recent WML client version, because that one
        # has 'get_details()' and I don't want to test the internals of
        # the client that I copied for older versions
        lib._client.version = "1.0.95.1"
        lib._client.data_assets = Mock(spec=Assets)
        lib._client.data_assets.create.return_value = {'metadata': {'guid': 'uid_for_created_asset'}}
        lib.inline = False
        domn = Mock()
        domn.SOLVE_PARAMETERS = 'solve-parameters'
        domn.INPUT_DATA_REFERENCES = 'input_data_references'
        domn.OUTPUT_DATA = 'output-data'
        lib._client.deployments.DecisionOptimizationMetaNames = domn
        lib.get_file_as_data = lambda path: 'base-64-content'
        lib._space_id = 'space-id'
        lib._get_deployment_id = Mock(spec=DOWMLLib._get_deployment_id)
        lib._get_deployment_id.return_value = 'deployment-id'
        self.lib = lib

    def test_solve_with_no_existing_data_asset_uploads_the_file(self):
        # Let's assume no existing asset
        self.lib._client.data_assets.get_details.return_value = {'resources': []}
        # and solve
        self.lib.solve('afiro.mps')
        # Check we've created the data asset
        data_asset_mock = self.lib._client.data_assets.create
        data_asset_mock.assert_called_once()
        # And we've used it
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(len(kall.args[1]['input_data_references']), 1)
        i = kall.args[1]['input_data_references'][0]
        self.assertEqual(i['location']['href'], '/v2/assets/uid_for_created_asset?space_id=space-id')

    def test_solve_with_existing_but_different_data_asset_uploads_the_file(self):
        # Let's assume no existing asset
        self.lib._client.data_assets.get_details.return_value = {'resources': [{'metadata': {'name': 'not_afiro.mps'}}]}
        # and solve
        self.lib.solve('afiro.mps')
        # Check we've created the data asset
        data_asset_mock = self.lib._client.data_assets.create
        data_asset_mock.assert_called_once()
        # And we've used it
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(len(kall.args[1]['input_data_references']), 1)
        i = kall.args[1]['input_data_references'][0]
        self.assertEqual(i['location']['href'], '/v2/assets/uid_for_created_asset?space_id=space-id')

    def test_solve_with_already_existing_data_asset_uploads_the_file(self):
        # Let's assume no existing asset
        self.lib._client.data_assets.get_details.return_value = {
            'resources': [
                {
                    'metadata': {
                        'name': 'afiro.mps',
                        'asset_id': 'id_for_afiro'
                    }
                }
            ]
        }
        # and solve
        self.lib.solve('afiro.mps')
        # Check we've NOT created the data asset
        data_asset_mock = self.lib._client.data_assets.create
        data_asset_mock.assert_not_called()
        # But we've used the existing one
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(len(kall.args[1]['input_data_references']), 1)
        i = kall.args[1]['input_data_references'][0]
        self.assertEqual(i['location']['href'], '/v2/assets/id_for_afiro?space_id=space-id')

    def test_solve_with_forced_update_uploads_the_file(self):
        # Let's assume no existing asset
        self.lib._client.data_assets.get_details.return_value = {
            'resources': [
                {
                    'metadata': {
                        'name': 'afiro.mps',
                        'asset_id': 'id_for_afiro'
                    }
                }
            ]
        }
        # and solve
        self.lib.solve('+afiro.mps')
        # Check we've created a new data asset
        create_data_asset_mock = self.lib._client.data_assets.create
        create_data_asset_mock.assert_called_once()
        # We've deleted the old one
        delete_data_asset_mock = self.lib._client.data_assets.delete
        delete_data_asset_mock.assert_called_once()
        delete_kall = delete_data_asset_mock.call_args
        self.assertEqual(delete_kall.args[0], 'id_for_afiro')
        # And used the new one
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(len(kall.args[1]['input_data_references']), 1)
        i = kall.args[1]['input_data_references'][0]
        self.assertEqual(i['location']['href'], '/v2/assets/uid_for_created_asset?space_id=space-id')


if __name__ == '__main__':
    main()
