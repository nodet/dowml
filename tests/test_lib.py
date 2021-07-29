import datetime
import pprint
from logging import Logger
from unittest.mock import Mock, patch

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.Set import Set
from ibm_watson_machine_learning.deployments import Deployments
from ibm_watson_machine_learning.assets import Assets
from ibm_watson_machine_learning.model_definition import ModelDefinition
from ibm_watson_machine_learning.spaces import Spaces

from dowmllib import DOWMLLib, SimilarNamesInJob, version_is_greater
from unittest import TestCase, main


TEST_CREDENTIALS_FILE_NAME = 'tests/test_credentials.txt'


class TestSolveInline(TestCase):

    def setUp(self) -> None:
        lib = DOWMLLib(TEST_CREDENTIALS_FILE_NAME)
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
        lib = DOWMLLib(TEST_CREDENTIALS_FILE_NAME)
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
        # Let's assume an asset already exists
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
        # And we'll succeed at deleting the old asset
        self.lib._client.data_assets.delete.return_value = "SUCCESS"
        # Now, solve
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


class TestGetJobs(TestCase):

    def setUp(self) -> None:
        lib = DOWMLLib(TEST_CREDENTIALS_FILE_NAME)
        lib.tz = datetime.timezone(datetime.timedelta(hours=2))
        lib._logger = Mock(spec=Logger)
        lib._client = Mock(spec=APIClient)
        lib._client.set = Mock(spec=Set)
        lib._client.deployments = Mock(spec=Deployments)
        lib._client.deployments.get_job_details.return_value = {'resources': []}
        lib._client.spaces = Mock(spec=Spaces)
        lib._client.spaces.get_details.return_value = {'resources': []}
        lib._client.spaces.get_uid.return_value = 'space_id'
        self.lib = lib

    def test_get_jobs_returns_empty_list_when_no_jobs(self):
        result = self.lib.get_jobs()
        self.assertListEqual(result, [])

    def test_get_jobs_parses_one_job_correctly_with_details_lacking(self):
        self.lib._client.deployments.get_job_details.return_value = {'resources': [
            # One job
            {
                'entity': {'decision_optimization': {'status': {'state': 'state'}}},
                'metadata': {
                    'id': 'id',
                    'created_at': '2021-05-02T15:58:02Z'
                }
            }
        ]}
        result = self.lib.get_jobs()
        # [Job(status='state', id='id', created='created_at', names=[], type='?????', version='?????', size='?')]
        self.assertEqual(len(result), 1)
        job = result[0]
        self.assertEqual(job.status, 'state')
        self.assertEqual(job.id, 'id')
        self.assertEqual(job.created, '2021-05-02 17:58:02')
        # The other details of the job were not set, we can assert that their
        # lacking has been dealt with correctly
        self.assertListEqual(job.names, [])
        self.assertEqual(job.type, '?????')
        self.assertEqual(job.version, '?????')
        self.assertEqual(job.size, '?')

    def test_get_jobs_parses_job_type_and_version_correctly(self):
        self.lib._client.deployments.get_details.return_value = {
            'entity': {
                'asset': {'id': 'model-id'},
                'hardware_spec': {'name': 'N'}
            }
        }
        self.lib._client.model_definitions = Mock(ModelDefinition)
        self.lib._client.model_definitions.get_details.return_value = {'entity': {'wml_model': {'type': 'do-cplex_1.0'}}}
        self.lib._client.deployments.get_job_details.return_value = {'resources': [
            # One job with a known deployment
            {
                'entity': {
                    'decision_optimization': {'status': {'state': 'state'}},
                    'deployment': {'id': 'deployment-id'}
                },
                'metadata': {
                    'id': 'id',
                    'created_at': 'created_at'
                }
            }
        ]}
        result = self.lib.get_jobs()
        self.lib._client.deployments.get_details.assert_called_once_with('deployment-id')
        self.lib._client.model_definitions.get_details.assert_called_once_with('model-id')
        j = result[0]
        self.assertEqual(j.type, 'cplex')
        self.assertEqual(j.version, '1.0')
        self.assertEqual(j.size, 'N')

    def test_get_jobs_parses_input_names(self):
        self.lib._client.deployments.get_job_details.return_value = {'resources': [
            # One job
            {
                'entity': {'decision_optimization': {
                    'status': {'state': 'state'},
                    'input_data': [{'id': 'foo'}],
                    'input_data_references': [
                        {'id': 'bar'},
                        {'unknown key': ''}
                    ]
                }},
                'metadata': {'id': 'id', 'created_at': 'created_at'}
            }
        ]}
        result = self.lib.get_jobs()
        # [Job(status='state', id='id', created='created_at', names=[], type='?????', version='?????', size='?')]
        self.assertEqual(len(result), 1)
        self.assertListEqual(result[0].names, ['foo', '*bar', 'Unknown'])


class TestWait(TestCase):

    def setUp(self) -> None:
        lib = DOWMLLib(TEST_CREDENTIALS_FILE_NAME)
        lib._logger = Mock(spec=Logger)
        lib._client = Mock(spec=APIClient)
        lib._client.set = Mock(spec=Set)
        lib._client.deployments = Mock(spec=Deployments)
        lib._client.spaces = Mock(spec=Spaces)
        lib._client.spaces.get_details.return_value = {'resources': []}
        self.lib = lib

    def test_wait_returns_after_one_call_if_job_is_already_completed(self):
        input_job_details = {
            'entity': {'decision_optimization': {
                'status': {'state': 'completed'}
            }},
            'metadata': {'id': 'job_id'}
        }
        self.lib._client.deployments.get_job_details.return_value = input_job_details
        status, job_details = self.lib.wait_for_job_end('job_id')
        self.lib._client.deployments.get_job_details.assert_called_once()
        self.assertEqual('completed', status)
        self.assertEqual(input_job_details, job_details)

    def test_wait_sleeps_as_long_as_job_not_complete_yet(self):
        not_complete_yet = {
            'entity': {'decision_optimization': {
                'status': {'state': 'running'}
            }},
            'metadata': {'id': 'job_id'}
        }
        # Number of times where the job will not yet be 'complete'
        nb_not_complete_calls = 3
        finished = {
            'entity': {'decision_optimization': {
                'status': {'state': 'completed'}
            }},
            'metadata': {'id': 'job_id'}
        }
        return_values = [not_complete_yet for i in range(nb_not_complete_calls)] + [finished]
        self.lib._client.deployments.get_job_details.side_effect = return_values
        with patch('time.sleep', return_value=None) as patched_time_sleep:
            _, _ = self.lib.wait_for_job_end('job_id')
        self.assertEqual(nb_not_complete_calls    , patched_time_sleep.call_count)
        self.assertEqual(nb_not_complete_calls + 1, self.lib._client.deployments.get_job_details.call_count)


class TestVersionComparison(TestCase):

    def test_1_0_95_greater_than_1_0_95(self):
        self.assertTrue(version_is_greater("1.0.95", "1.0.95"))

    def test_1_0_95_greater_than_1_0_94(self):
        self.assertTrue(version_is_greater("1.0.95", "1.0.94"))

    def test_1_0_94_not_greater_than_1_0_95(self):
        self.assertFalse(version_is_greater("1.0.94", "1.0.95"))

    def test_1_0_100_greater_than_1_0_95(self):
        self.assertTrue(version_is_greater("1.0.100", "1.0.95"))

    def test_1_0_100_1_greater_than_1_0_100(self):
        self.assertTrue(version_is_greater("1.0.100.1", "1.0.100"))


class TestFailingToCheckCIWorkflow(TestCase):

    def test_fails(self):
        self.fail('Failing test, to check what happens in CI')


if __name__ == '__main__':
    main()
