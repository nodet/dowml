import datetime
import os
import sys
from logging import Logger
from unittest.mock import Mock, patch, call

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.Set import Set
from ibm_watson_machine_learning.deployments import Deployments
from ibm_watson_machine_learning.assets import Assets
from ibm_watson_machine_learning.model_definition import ModelDefinition
from ibm_watson_machine_learning.spaces import Spaces

from unittest import TestCase, main, mock

from ibm_watson_machine_learning.wml_client_error import WMLClientError, ApiRequestFailure

from dowml.lib import InvalidCredentials, _CredentialsProvider, DOWMLLib, SimilarNamesInJob, version_is_greater

TEST_CREDENTIALS_FILE_NAME = 'test_credentials.txt'


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class TestCredentials(TestCase):

    def test_empty_credentials_raises_error(self):
        with self.assertRaises(InvalidCredentials):
            _ = _CredentialsProvider(wml_credentials_str='')

    def test_empty_url_raises_error(self):
        with self.assertRaises(InvalidCredentials):
            _ = _CredentialsProvider(wml_credentials_str='{\'apikey\': \'<apikey>\', \'url\': \'\'}')

    def test_space_name_has_default(self):
        with mock.patch.dict(os.environ, {
                'DOWML_CREDENTIALS': "{'apikey': '<apikey>', 'url': 'https://us-south.ml.cloud.ibm.com'}"}):
            default_space_name = 'dowml-space'
            # Let's check the default value is correct
            lib = DOWMLLib()
            self.assertEqual(default_space_name, lib.space_name)

    def test_space_name_in_credentials_change_default(self):
        # Let's now try to change that default value
        non_default_name = 'dowml-test-space'
        lib = DOWMLLib(TEST_CREDENTIALS_FILE_NAME)
        # Let's first check that the test was setup properly: it relies on
        # 'space-name' being included in the test credentials file
        # FIXME: the test would be much cleaner if DOWMLlib would accept credentials directly
        self.assertIn(_CredentialsProvider.SPACE_NAME, lib._wml_credentials)
        self.assertEqual(non_default_name, lib._wml_credentials[_CredentialsProvider.SPACE_NAME])
        # Let's confirm that the space name is changed
        self.assertEqual(lib.space_name, non_default_name)

    def test_space_id_in_constructor_overrides_credentials(self):
        # Let's now try to change that default value
        non_default_space_id = 'foo'
        # Let's make sure we are really testing something: by default, we don't
        # have a space_id in the credentials
        with mock.patch.dict(os.environ, {
            'DOWML_CREDENTIALS': "{'apikey': '<apikey>',"
                                 " 'url': 'https://us-south.ml.cloud.ibm.com',"
                                 " 'space_id': 'bar'}"}):
            lib = DOWMLLib()
            self.assertNotEqual(non_default_space_id, lib._wml_credentials['space_id'])
            # And now we change the default
            lib = DOWMLLib(space_id=non_default_space_id)
        # Let's confirm that the space name is changed
        self.assertEqual(non_default_space_id, lib._wml_credentials['space_id'])

    def test_url_with_trailing_slash_is_accounted_for_automatically(self):
        url = 'https://us-south.ml.cloud.ibm.com/'
        wml_credentials_str = '{\'apikey\': \'<apikey>\', \'url\': \'https://us-south.ml.cloud.ibm.com/\'}'
        cred_provider = _CredentialsProvider(wml_credentials_str=wml_credentials_str)
        self.assertEqual(url[:-1], cred_provider.credentials['url'])

    def test_error_if_no_credentials_found_anywhere(self):
        with mock.patch.dict(os.environ, clear=True):
            with self.assertRaises(InvalidCredentials):
                _ = DOWMLLib()

    def test_lib_reads_file_referenced_in_env_var(self):
        # Let's make sure we are really testing something: clear the environment
        with mock.patch.dict(os.environ, clear=True):
            with self.assertRaises(InvalidCredentials):
                _ = DOWMLLib()
            # We did get the exception when no credentials were found
            # Let's check we find some with the new environment variable
            os.environ['DOWML_CREDENTIALS_FILE'] = TEST_CREDENTIALS_FILE_NAME
            _ = DOWMLLib()

    def test_url_in_constructor_overrides_credentials(self):
        default_url = 'https://cloud.ibm.com'
        non_default_url = 'https://non.default.url.ibm.com'
        with mock.patch.dict(os.environ,
                             {'DOWML_CREDENTIALS': f"{{'apikey': '<apikey>', 'url': '{default_url}'}}"}
                             ):
            lib = DOWMLLib()
            self.assertEqual(default_url, lib._wml_credentials['url'])
            # And now we change the default
            lib = DOWMLLib(url=non_default_url)
            self.assertEqual(non_default_url, lib._wml_credentials['url'])

    def test_region_in_credentials_define_a_url(self):
        with mock.patch.dict(os.environ,
                             {'DOWML_CREDENTIALS': "{'apikey': '<apikey>', 'region': 'us-south'}"}):
            lib = DOWMLLib()
            self.assertEqual('https://us-south.ml.cloud.ibm.com', lib._wml_credentials['url'])

    def test_region_is_removed_from_credentials(self):
        with mock.patch.dict(os.environ,
                             {'DOWML_CREDENTIALS': "{'apikey': '<apikey>', 'region': 'eu-de'}"}):
            lib = DOWMLLib()
            self.assertNotIn('region', lib._wml_credentials)

    def test_error_if_unkown_region(self):
        with mock.patch.dict(os.environ,
                             {'DOWML_CREDENTIALS': "{'apikey': '<apikey>', 'region': 'unknown'}"}):
            with self.assertRaises(InvalidCredentials):
                _ = DOWMLLib()

    def test_error_if_both_region_and_url_are_specified(self):
        with mock.patch.dict(os.environ, {
            'DOWML_CREDENTIALS': "{'apikey': '<apikey>', "
                                 "'url': 'https://us-south.ml.cloud.ibm.com', 'region': 'eu-gb'}"}):
            with self.assertRaises(InvalidCredentials):
                _ = DOWMLLib()

    def test_region_in_constructor_overrides_credentials(self):
        default_url = 'htts://cloud.ibm.com'
        with mock.patch.dict(os.environ,
                             {'DOWML_CREDENTIALS': f"{{'apikey': '<apikey>', 'url': '{default_url}'}}"}
                             ):
            lib = DOWMLLib()
            self.assertEqual(default_url, lib._wml_credentials['url'])
            # And now we change the default
            lib = DOWMLLib(region='jp-tok')
            self.assertEqual('https://jp-tok.ml.cloud.ibm.com', lib._wml_credentials['url'])

    def test_region_and_url_not_both_in_constructor(self):
        with self.assertRaises(InvalidCredentials):
            _ = DOWMLLib(url='https://jp-tok.ml.cloud.ibm.com', region='eu-de')

    def test_can_specify_region_on_top_of_no_url(self):
        """The point of this test is to check GH-41: when credentials don't include
        a URL, specifying a region in the constructor would crash"""
        with mock.patch.dict(os.environ,
                             {'DOWML_CREDENTIALS': "{'apikey': '<apikey>', 'region': 'eu-de'}"}):
            _ = DOWMLLib(region='eu-gb')


class TestLibAttributes(TestCase):
    URL = 'the-url'

    def test_lib_has_url(self):
        lib = DOWMLLib(url=self.URL)
        self.assertEqual(self.URL, lib.url)

    def test_url_is_readonly(self):
        lib = DOWMLLib(url=self.URL)
        with self.assertRaises(AttributeError):
            lib.url = 'the-new-url'


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
        self.assertEqual(params['oaas.timeLimit'], 1000 * lim)

    def test_solve_defaults_with_zero_limit(self):
        self.lib.timelimit = 0
        self.lib.solve('path')
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        self.assertNotIn('oaas.timeLimit', self.get_params(create_job_mock))

    def test_deprecated_get_output(self):
        details = {'entity': {'decision_optimization': {}}}
        _ = self.lib.get_output(details)
        self.assertTrue(True)

    def test_get_no_output(self):
        details = {'entity': {'decision_optimization': {}}}
        output = self.lib.get_outputs(details)
        self.assertIsInstance(output, dict)
        self.assertEqual(0, len(output))

    def test_get_csv_file(self):
        details = {'entity': {'decision_optimization': {'output_data': [{
            'fields': ['i', 'f'],
            'id': 'results.csv',
            'values': [
                [0, 0],
                ['a,b', 'c d']
            ]
        }]}}}
        # We use the deprecated parameter, to confirm it's still there
        output = self.lib.get_outputs(details, csv_as_dataframe=False)
        self.assertIsInstance(output, dict)
        self.assertEqual(1, len(output))
        self.assertIn('results.csv', output)
        lines = output['results.csv'].decode().split('\r\n')
        # There is one empty line at the end
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], 'i,f')
        self.assertEqual(lines[1], '0,0')
        self.assertEqual(lines[2], '"a,b",c d')

    def test_get_csv_file_the_new_way(self):
        details = {'entity': {'decision_optimization': {'output_data': [{
            'fields': ['i', 'f'],
            'id': 'results.csv',
            'values': [
                [0, 0],
                ['a,b', 'c d']
            ]
        }]}}}
        output = self.lib.get_outputs(details, tabular_as_csv=True)
        self.assertEqual(type(b''), type(output['results.csv']))

    def test_get_regular_output_file(self):
        name = 'one-var.lp'
        details = {'entity': {
            'decision_optimization': {
                'output_data': [{
                    'id': name,
                    'content': 'bWluaW1pemUgeApzdAogICB4ID49IDIKZW5k'
                }]
            }
        }}
        output = self.lib.get_outputs(details)
        self.assertIsInstance(output, dict)
        self.assertEqual(1, len(output))
        self.assertIn(name, output)
        lines = output[name].decode().split('\n')
        self.assertEqual('minimize x', lines[0])

    def test_get_regular_input_file(self):
        name = 'one-var.lp'
        details = {'entity': {
            'decision_optimization': {
                'input_data': [{
                    'id': name,
                    'content': 'bWluaW1pemUgeApzdAogICB4ID49IDIKZW5k'
                }]
            }
        }}
        output = self.lib.get_inputs(details)
        self.assertIsInstance(output, dict)
        self.assertEqual(1, len(output))
        self.assertIn(name, output)
        lines = output[name].decode().split('\n')
        self.assertEqual('minimize x', lines[0])

    def test_get_unknown_file(self):
        name = 'unknown'
        content = {
            'id': name,
            'foo': 'bar'
        }
        details = {
            'entity': {'decision_optimization': {'output_data': [content]}}
        }
        output = self.lib.get_outputs(details)
        # We get back the raw content
        self.assertEqual(content, output[name])


class TestSolveCachesDeploymentInformation(TestCase):

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
        lib._get_deployment_id_with_params_cached = Mock(spec=DOWMLLib._get_deployment_id_with_params_cached)
        lib._get_deployment_id_with_params_cached.return_value = 'deployment-id'
        self.lib = lib

    def test_solve_checks_deployment_info(self):
        self.lib.solve('afiro.mps')
        self.lib._get_deployment_id_with_params_cached.assert_called_once()

    def test_solve_checks_deployment_info_only_once(self):
        self.lib.solve('afiro.mps')
        self.lib.solve('afiro.mps')
        self.lib._get_deployment_id_with_params_cached.assert_called_once()

    def test_solve_checks_deployment_info_again_if_failure(self):
        mock_get_id = self.lib._get_deployment_id_with_params_cached
        mock_get_id.side_effect = ['stale-deployment-id', 'new-deployment-id']
        create_job_mock = self.lib._client.deployments.create_job
        mock_response = Mock()
        mock_response.status_code = ''
        mock_response.content = b'deployment_does_not_exist'
        with HiddenPrints():
            create_job_mock.side_effect = [ApiRequestFailure('error message', mock_response), 'job-id']
        self.lib.solve('afiro.mps')
        self.assertEqual(2, mock_get_id.call_count)

    def test_solve_checks_deployment_again_when_job_type_changes(self):
        self.lib.model_type = 'cplex'
        self.lib.solve('afiro.mps')
        self.lib.model_type = 'docplex'
        self.lib.solve('foo.py')
        self.assertEqual(2, self.lib._get_deployment_id_with_params_cached.call_count)


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
        # Let's assume we have an existing asset
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


class TestOutputs(TestCase):

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
        domn.OUTPUT_DATA_REFERENCES = 'output_data_references'
        lib._client.deployments.DecisionOptimizationMetaNames = domn
        lib.get_file_as_data = lambda path: 'base-64-content'
        lib._space_id = 'space-id'
        lib._get_deployment_id = Mock(spec=DOWMLLib._get_deployment_id)
        lib._get_deployment_id.return_value = 'deployment-id'
        self.lib = lib

    def test_solve_with_outputs_inline(self):
        # If outputs are inline, we create a single output-data,
        # that catches all outputs
        self.assertEqual('inline', self.lib.outputs)
        self.lib.solve('afiro.mps')
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(kall.kwargs, {})
        self.assertEqual(kall.args[0], 'deployment-id')
        self.assertEqual(len(kall.args[1]['output-data']), 1)
        i = kall.args[1]['output-data'][0]
        self.assertEqual(i['id'], '.*')

    def test_solve_with_outputs_assets(self):
        # If outputs are assets, we create a single output-data-reference,
        # that catches all outputs
        self.lib.outputs = 'assets'
        self.lib.solve('afiro.mps')
        create_job_mock = self.lib._client.deployments.create_job
        create_job_mock.assert_called_once()
        kall = create_job_mock.call_args
        self.assertEqual(kall.kwargs, {})
        self.assertEqual(kall.args[0], 'deployment-id')
        odr = 'output_data_references'
        self.assertEqual(len(kall.args[1][odr]), 1)
        i = kall.args[1][odr][0]
        self.assertEqual('data_asset', i['type'])
        self.assertEqual('.*', i['id'])
        self.assertEqual({}, i['connection'])
        self.assertEqual({'name': '${job_id}/${attachment_name}'}, i['location'])


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
        self.lib._client.model_definitions.get_details.return_value = {'entity': {
            'wml_model': {'type': 'do-cplex_1.0'}}
        }
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
        lib._client.version = "0.0.0"
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
        status, job_details = self.lib.wait_for_job_end('job_id', print_activity=True)
        self.lib._client.deployments.get_job_details.assert_called_once()
        self.assertEqual('completed', status)
        self.assertEqual(input_job_details, job_details)

    def test_wait_sleeps_as_long_as_job_not_complete_yet(self):
        no_status_yet = {
            'entity': {'decision_optimization': {
                'status': {'state': 'queued'},
            }},
            'metadata': {'id': 'job_id'}
        }
        not_complete_yet = {
            'entity': {'decision_optimization': {
                'status': {'state': 'running'},
                'solve_state': {'latest_engine_activity': ['foo', 'bar']}
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
        return_values = [no_status_yet] + [not_complete_yet for _ in range(nb_not_complete_calls)] + [finished]
        self.lib._client.deployments.get_job_details.side_effect = return_values
        with patch('time.sleep', return_value=None) as patched_time_sleep:
            _, _ = self.lib.wait_for_job_end('job_id', print_activity=True)
        self.assertEqual(nb_not_complete_calls + 1, patched_time_sleep.call_count)
        self.assertEqual(nb_not_complete_calls + 2, self.lib._client.deployments.get_job_details.call_count)


class TestLog(TestCase):

    def setUp(self) -> None:
        lib = DOWMLLib(TEST_CREDENTIALS_FILE_NAME)
        lib._logger = Mock(spec=Logger)
        lib._client = Mock(spec=APIClient)
        lib._client.version = "0.0.0"
        lib._client.set = Mock(spec=Set)
        lib._client.deployments = Mock(spec=Deployments)
        lib._client.spaces = Mock(spec=Spaces)
        lib._client.spaces.get_details.return_value = {'resources': []}
        self.lib = lib

    def test_get_log_return_none_if_no_do_structure(self):
        self.lib._client.deployments.get_job_details.return_value = {}
        self.assertIsNone(self.lib.get_log('1'))

    def test_get_log_return_none_if_no_output(self):
        self.lib._client.deployments.get_job_details.return_value = {'entity': {'decision_optimization': {}}}
        self.assertIsNone(self.lib.get_log('1'))

    def test_get_log_return_none_if_empty_log(self):
        self.lib._client.deployments.get_job_details.return_value = {'entity': {'decision_optimization': {
            'output_data': [{'id': 'log.txt'}]
        }}}
        self.assertIsNone(self.lib.get_log('1'))

    def test_get_log_removes_empty_lines(self):
        self.lib._client.deployments.get_job_details.return_value = {'entity': {'decision_optimization': {
            'output_data': [{
                'id': 'log.txt',
                # This is 'line 1\n\nline 2\n' encoded with
                #    openssl base64 < log.txt
                'content': 'bGluZSAxCgpsaW5lIDIK'
            }]
        }}}
        # Let's confirm that empty lines were removed
        self.assertEqual('line 1\nline 2', self.lib.get_log('1'))

    def test_get_log_fetches_asset_if_necessary(self):
        self.lib._client.deployments.get_job_details.return_value = {'entity': {'decision_optimization': {
            'output_data_references': [
                {
                    'type': 'unknown'
                },
                {
                    'type': 'data_asset'
                },
                {
                    'id': 'log.txt',
                    'location': {
                        'id': 'id1'
                    },
                    'type': 'data_asset'
                }
            ]
        }}}
        self.lib._client.data_assets = Mock(spec=Assets)
        self.lib._client.data_assets.download.return_value = 'path'
        content = 'content of log'
        with mock.patch('builtins.open', mock.mock_open(read_data=content)):
            self.assertEqual(content, self.lib.get_log('1'))

    def test_get_log_stops_if_log_asset_misses_location(self):
        self.lib._client.deployments.get_job_details.return_value = {'entity': {'decision_optimization': {
            'output_data_references': [
                {
                    'id': 'log.txt',
                    'type': 'data_asset'
                }
            ]
        }}}
        self.assertIsNone(self.lib.get_log('1'))

    def test_get_log_stops_if_log_asset_misses_location_id(self):
        self.lib._client.deployments.get_job_details.return_value = {'entity': {'decision_optimization': {
            'output_data_references': [
                {
                    'id': 'log.txt',
                    'location': {},
                    'type': 'data_asset'
                }
            ]
        }}}
        self.assertIsNone(self.lib.get_log('1'))

    def test_get_log_deals_with_not_finding_log_asset(self):
        self.lib._client.deployments.get_job_details.return_value = {'entity': {'decision_optimization': {
            'output_data_references': [
                {}
            ]
        }}}
        self.assertIsNone(self.lib.get_log('1'))


class TestDeleteJob(TestCase):

    def setUp(self) -> None:
        lib = DOWMLLib(TEST_CREDENTIALS_FILE_NAME)
        lib._logger = Mock(spec=Logger)
        lib._client = Mock(spec=APIClient)
        lib._client.set = Mock(spec=Set)
        lib._client.deployments = Mock(spec=Deployments)
        lib._client.spaces = Mock(spec=Spaces)
        lib._client.data_assets = Mock(spec=Assets)
        lib._client.spaces.get_details.return_value = {'resources': []}
        lib._client.PLATFORM_URLS_MAP = {'https://eu-de.ml.cloud.ibm.com': 'https://the.ws.url.ibm.com'}
        lib._wml_credentials['url'] = 'https://eu-de.ml.cloud.ibm.com'
        lib.get_job_details = Mock()
        lib.get_job_details.return_value = {}
        self.lib = lib

    def test_delete_soft_doesnt_look_for_job_details(self):
        job_id = 'job_id'
        self.lib.delete_job(job_id, hard=False)
        self.lib._client.deployments.delete_job.assert_called_once_with(job_id, False)
        self.lib.get_job_details.assert_not_called()

    def test_delete_hard_doesnt_delete_assets_if_job_has_none(self):
        job_id = 'job_id'
        self.lib.get_job_details.return_value = {
            'entity': {'decision_optimization': {
                'output_data_references': []
            }},
            'metadata': {
                'id': job_id
            },
        }
        self.lib.delete_job(job_id, hard=True)
        self.lib._client.deployments.delete_job.assert_called_once_with(job_id, True)
        self.lib.get_job_details.assert_called_once_with(job_id, with_contents='names')
        self.lib._client.data_assets.delete.assert_not_called()

    def test_delete_hard_does_delete_assets_if_job_has_any(self):
        job_id = 'job_id'
        data_asset_id = 'data_asset_id'
        self.lib.get_job_details.return_value = {
            'entity': {'decision_optimization': {
                'output_data_references': [
                    # One output that is not a data_asset
                    {'type': 'connection_asset'},
                    # Error case: A data-asset that has no location
                    {'type': 'data_asset'},
                    # Error case: A data-asset that has no id
                    {
                        'location': {},
                        'type': 'data_asset'
                    },
                    # A 'normal' data asset output
                    {
                        'location': {'id': data_asset_id},
                        'type': 'data_asset'
                    },
                ]
            }},
            'metadata': {
                'id': job_id
            },
        }
        delete_job = self.lib._client.deployments.delete_job
        get_job_details = self.lib.get_job_details
        delete_asset = self.lib._client.data_assets.delete

        # This mock will record the call orders of its children
        m = Mock()
        m.attach_mock(delete_job, 'delete_job')
        m.attach_mock(get_job_details, 'get_job_details')
        m.attach_mock(delete_asset, 'delete_asset')

        self.lib.delete_job(job_id, hard=True)
        delete_job.assert_called_once_with(job_id, True)
        # By default, get_job_details will filter the outputs, we need them
        # when we want to check what data asset to delete
        get_job_details.assert_called_once_with(job_id, with_contents='names')
        delete_asset.assert_called_once_with('data_asset_id')
        # Delete the assets first, and only then delete the job itself
        m.assert_has_calls([
            call.get_job_details(job_id, with_contents='names'),
            call.delete_asset(data_asset_id),
            call.delete_job(job_id, True)
        ])

    def test_delete_catches_errors_for_missing_assets(self):
        job_id = 'job_id'
        self.lib.get_job_details.return_value = {
            'entity': {'decision_optimization': {
                'output_data_references': [
                    {
                        'location': {'id': 'an_id'},
                        'type': 'data_asset'
                    },
                    # Another one, to confirm that the error doesn't
                    # stop the loop
                    {
                        'location': {'id': 'another_id'},
                        'type': 'data_asset'
                    },
                ]
            }},
            'metadata': {
                'id': job_id
            },
        }
        delete_asset = self.lib._client.data_assets.delete
        delete_asset.side_effect = WMLClientError("delete assets failed")
        self.lib.delete_job(job_id, hard=True)
        self.assertEqual(2, delete_asset.call_count)

    def test_delete_deletes_the_platform_job(self):
        self.lib._client.service_instance = Mock()
        job_id = 'job_id'
        self.lib.get_job_details.return_value = {
            'entity': {
                'platform_job': {
                    'job_id': 'platform-job-id',
                    'run_id': 'platform-run-id'
                }
            },
            'metadata': {
                'id': job_id
            },
        }
        mock_response = Mock()
        mock_response.status_code = 204
        mock_delete = Mock()
        mock_delete.return_value = mock_response
        with mock.patch('requests.delete', mock_delete):
            self.lib.delete_job(job_id, hard=True)
        mock_delete.assert_called_once()
        args, _ = mock_delete.call_args_list[0]
        self.assertEqual('https://the.ws.url.ibm.com/v2/jobs/platform-job-id/runs/platform-run-id?space_id=None',
                         args[0])

    def test_delete_deals_with_absent_platform_job_information(self):
        job_id = 'job_id'
        self.lib.get_job_details.return_value = {
            # But we don't give any info about a platform job
            'metadata': {
                'id': job_id
            },
        }
        self.lib.delete_job(job_id, hard=True)
        self.lib._client.deployments.delete_job.assert_called_once_with(job_id, True)

    def test_delete_deals_with_error_when_deleting_platform_job(self):
        self.lib._client.service_instance = Mock()
        job_id = 'job_id'
        self.lib.get_job_details.return_value = {
            'entity': {
                'platform_job': {
                    'job_id': 'platform-job-id',
                    'run_id': 'platform-run-id'
                }
            },
            'metadata': {
                'id': job_id
            },
        }
        mock_response = Mock()
        mock_response.status_code = 404
        mock_delete = Mock()
        mock_delete.return_value = mock_response
        with mock.patch('requests.delete', mock_delete):
            self.lib.delete_job(job_id, hard=True)
        self.lib._client.deployments.delete_job.assert_called_once_with(job_id, True)


class TestInputAndOutputGathering(TestCase):

    def setUp(self) -> None:
        lib = DOWMLLib(TEST_CREDENTIALS_FILE_NAME)
        lib.tz = datetime.timezone(datetime.timedelta(hours=2))
        lib._logger = Mock(spec=Logger)
        lib._client = Mock(spec=APIClient)
        self.lib = lib

    def test_get_output_data_ref_ignores_non_assets(self):
        result = self.lib.get_output_asset_ids({'entity': {'decision_optimization': {
            'output_data_references': [
                {'type': 'unknown'},
                {},
                {
                    # No name for this data-asset
                    'type': 'data_asset'
                },
                {
                    'id': 'log.txt',
                    'location': {'id': 'id1'},
                    'type': 'data_asset'
                }
            ]
        }}})
        self.assertDictEqual(result, {'log.txt': 'id1'})

    def test_get_output_data_ref_deals_with_lack_of_info(self):
        result = self.lib.get_output_asset_ids({'entity': {'decision_optimization': {}}})
        self.assertDictEqual(result, {})
        result = self.lib.get_output_asset_ids({'entity': {}})
        self.assertDictEqual(result, {})
        result = self.lib.get_output_asset_ids({})
        self.assertDictEqual(result, {})

    def test_get_input_data_references(self):
        result = self.lib.get_input_asset_ids({'entity': {'decision_optimization': {
            'input_data_references': [
                {
                    'id': 'afiro.mps',
                    # We can handle ids alone
                    'location': {'id': 'id1'},
                    'type': 'data_asset'
                },
                {
                    'id': 'foo.prm',
                    # We can handle hrefs alone
                    'location': {'href': '/v2/assets/asset-id?space_id=space-id'},
                    'type': 'data_asset'
                },
                {
                    'id': 'bar',
                    # When faced with both, id is used
                    'location': {
                        'href': '/v2/assets/asset-id?space_id=space-id',
                        'id': 'the-real-asset-id'
                    },
                    'type': 'data_asset'
                },
                {
                    'id': 'baz',
                    # We ignore assets which we can't locate
                    'location': {},
                    'type': 'data_asset'
                },
                {
                    'id': 'bazz',
                    # This one will be ignored because of incorrect href
                    'location': {'href': 'bogus'},
                    'type': 'data_asset'
                },
                {
                    'id': 'bazzz',
                    # This one doesn't even have a location
                    'type': 'data_asset'
                }
            ]
        }}})
        self.assertDictEqual(result, {
            'afiro.mps': 'id1',
            'foo.prm': 'asset-id',
            'bar': 'the-real-asset-id'
        })


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


if __name__ == '__main__':
    main()
