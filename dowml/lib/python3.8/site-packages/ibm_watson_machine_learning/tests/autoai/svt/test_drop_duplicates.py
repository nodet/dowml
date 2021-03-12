import unittest
from sklearn.pipeline import Pipeline
from pprint import pprint
import pandas as pd
import traceback
from os import environ

from ibm_watson_machine_learning import APIClient

from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import WebService, Batch
from ibm_watson_machine_learning.workspace import WorkSpace

from ibm_watson_machine_learning.helpers.connections import S3Connection, S3Location, DataConnection, FSLocation

from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines

from ibm_watson_machine_learning.utils.autoai.errors import NotExistingCOSResource

from ibm_watson_machine_learning.tests.utils import (get_wml_credentials, get_cos_credentials, bucket_exists,
                                                        bucket_name_gen, get_space_id, is_cp4d)
from ibm_watson_machine_learning.tests.utils.cleanup import bucket_cleanup, space_cleanup, delete_model_deployment

from ibm_watson_machine_learning.utils.autoai.enums import PositiveLabelClass, TShirtSize, Transformers


from lale import wrap_imported_operators
from lale.operators import TrainablePipeline
from lale.lib.lale import Hyperopt


class TestAutoAIRemote(unittest.TestCase):
    """
    The test can be run on CLOUD, WMLS and CPD (not tested)
    The test covers:
    - COS set-up (if run on Cloud): checking if bucket exists for the cos instance, if not new bucket is create
    - Saving data `/bank.cdv` to COS/data assets
    - downloading training data from cos/data assets
    - downloading all generated pipelines to lale pipeline
    - deployment with lale pipeline
    - deployment deletion
    """
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None
    pipeline_opt: 'RemoteAutoPipelines' = None
    service: 'WebService' = None
    service_batch: 'Batch' = None

    data_location = './autoai/data/iris_dataset.csv'

    prediction_column = 'species'

    trained_pipeline_details = None
    run_id = None

    data_connection = None
    results_connection = None

    train_data = None

    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa")
    pod_version = environ.get('KB_VERSION', None)

    cos_endpoint = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
    data_cos_path = 'data/iris_dataset.csv'
    cos_resource_instance_id = None

    results_cos_path = 'results_wml_autoai'

    pipeline: 'Pipeline' = None
    lale_pipeline = None
    deployed_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None
    X_values = None
    y_values = None

    OPTIMIZER_NAME = 'IRIS csv wml_autoai multiclass test'
    DEPLOYMENT_NAME = "IRIS AutoAI Deployment tests"

    project_id = None
    space_id = None

    space_name = 'tests_sdk_space'

    asset_id = None

    row_no_with_drop = None
    row_no_without_drop = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.wml_credentials = get_wml_credentials()
        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials.copy())

        if not cls.wml_client.ICP:
            cls.cos_credentials = get_cos_credentials()
            cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
            cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.project_id = cls.wml_credentials.get('project_id')


    def test_00a_space_cleanup(self):
        space_cleanup(self.wml_client,
                      get_space_id(self.wml_client, self.space_name,
                                   cos_resource_instance_id=self.cos_resource_instance_id),
                      days_old=7)
        TestAutoAIRemote.space_id = get_space_id(self.wml_client, self.space_name,
                                     cos_resource_instance_id=self.cos_resource_instance_id)
        if self.wml_client.ICP:
            self.wml_client.set.default_project(self.project_id)
        else:
            self.wml_client.set.default_space(self.space_id)


    def test_00b_prepare_COS_instance(self):
        if self.wml_client.ICP:
            self.skipTest("Prepare COS is available only for Cloud")
        import ibm_boto3
        cos_resource = ibm_boto3.resource(
            service_name="s3",
            endpoint_url=self.cos_endpoint,
            aws_access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
            aws_secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
        )
        # Prepare bucket
        if not bucket_exists(cos_resource, self.bucket_name):
            try:
                bucket_cleanup(cos_resource, prefix=f"{self.bucket_name}-")
            except Exception as e:
                print(f"Bucket cleanup with prefix {self.bucket_name}- failed due to:\n{e}\n skipped")

            import datetime
            TestAutoAIRemote.bucket_name = bucket_name_gen(prefix=f"{self.bucket_name}-{str(datetime.date.today())}")
            print(f"Creating COS bucket: {TestAutoAIRemote.bucket_name}")
            cos_resource.Bucket(TestAutoAIRemote.bucket_name).create()

            self.assertIsNotNone(TestAutoAIRemote.bucket_name)
            self.assertTrue(bucket_exists(cos_resource, TestAutoAIRemote.bucket_name))

        print(f"Using COS bucket: {TestAutoAIRemote.bucket_name}")

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                             project_id=self.project_id,
                                             space_id=self.space_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_02_save_remote_data_and_DataConnection_setup(self):
        if self.wml_client.ICP:
            TestAutoAIRemote.data_connection = DataConnection(
                            location=FSLocation(path=self.data_location))
            TestAutoAIRemote.results_connection = None

        else: #for cloud and COS
            TestAutoAIRemote.data_connection = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                        secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path)
            )
            TestAutoAIRemote.results_connection = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                        secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.results_cos_path)
            )
            TestAutoAIRemote.data_connection.write(data=self.data_location, remote_name=self.data_cos_path)

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)

    def test_02a_read_saved_remote_data_before_fit(self):
        data = self.data_connection.read()
        print(f"Data sample")
        print(data.head())
        self.assertGreater(len(data), 0)

    def test_03_initialize_optimizer_with_drop_duplicates(self):
        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            desc='test description',
            prediction_type=self.experiment.PredictionType.MULTICLASS,
            prediction_column=self.prediction_column,
            scoring=PositiveLabelClass.PRECISION_SCORE_MICRO,
            cognito_transform_names=[Transformers.DIFF,Transformers.SUM,Transformers.MAX,
                                     Transformers.SIN,Transformers.SUM],
            test_size=0.1,
            max_number_of_estimators=1,
            t_shirt_size=TShirtSize.S if self.wml_client.ICP else TShirtSize.L
        )

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_04_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)
        assert ('drop_duplicates' in parameters)
        assert (parameters['drop_duplicates'])
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_05_fit_run_training_of_auto_ai_in_wml(self):
        TestAutoAIRemote.trained_pipeline_details = self.remote_auto_pipelines.fit(
            training_data_reference=[self.data_connection],
            training_results_reference=self.results_connection,
            background_mode=False)

        TestAutoAIRemote.run_id = self.trained_pipeline_details['metadata']['id']

        self.assertIsNotNone(self.data_connection.auto_pipeline_params,
                             msg='DataConnection auto_pipeline_params was not updated.')

        step = self.trained_pipeline_details['entity']['status']['metrics'][0]['context']['intermediate_model']['composition_steps'][0]
        TestAutoAIRemote.row_no_with_drop = int(step.split('_')[2])
        print(TestAutoAIRemote.row_no_with_drop)
        self.assertEqual(TestAutoAIRemote.row_no_with_drop, 147)

    def test_06_initialize_optimizer_without_drop_duplicates(self):
        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            desc='test description',
            prediction_type=self.experiment.PredictionType.MULTICLASS,
            prediction_column=self.prediction_column,
            scoring=PositiveLabelClass.PRECISION_SCORE_MICRO,
            cognito_transform_names=[Transformers.DIFF,Transformers.SUM,Transformers.MAX,
                                     Transformers.SIN,Transformers.SUM],
            test_size=0.1,
            max_number_of_estimators=1,
            t_shirt_size=TShirtSize.S if self.wml_client.ICP else TShirtSize.L,
            drop_duplicates=False
        )

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_07_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)
        assert('drop_duplicates' in parameters)
        assert(not parameters['drop_duplicates'])
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_08_fit_run_training_of_auto_ai_in_wml(self):
        TestAutoAIRemote.trained_pipeline_details = self.remote_auto_pipelines.fit(
            training_data_reference=[self.data_connection],
            training_results_reference=self.results_connection,
            background_mode=False)

        TestAutoAIRemote.run_id = self.trained_pipeline_details['metadata']['id']

        self.assertIsNotNone(self.data_connection.auto_pipeline_params,
                             msg='DataConnection auto_pipeline_params was not updated.')

        step = self.trained_pipeline_details['entity']['status']['metrics'][0]['context']['intermediate_model']['composition_steps'][0]
        TestAutoAIRemote.row_no_without_drop = int(step.split('_')[2])
        print(TestAutoAIRemote.row_no_without_drop)
        self.assertEqual(TestAutoAIRemote.row_no_without_drop, 150)

    def test_09_compare(self):
        print(TestAutoAIRemote.row_no_with_drop, TestAutoAIRemote.row_no_without_drop)
        self.assertNotEqual(TestAutoAIRemote.row_no_with_drop, TestAutoAIRemote.row_no_without_drop)

if __name__ == '__main__':
    unittest.main()
