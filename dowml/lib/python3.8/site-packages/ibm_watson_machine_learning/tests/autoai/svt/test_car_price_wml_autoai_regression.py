import unittest
import pandas as pd
from sklearn.pipeline import Pipeline
from lale.operators import TrainablePipeline
from lale.lib.lale import Hyperopt
from os import environ
import time


import traceback
import pprint

from ibm_watson_machine_learning import APIClient
from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.helpers.connections import S3Connection, S3Location, DataConnection, FSLocation,\
    AssetLocation, DeploymentOutputAssetLocation
from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.deployment import WebService, Batch
from ibm_watson_machine_learning.workspace import WorkSpace


from ibm_watson_machine_learning.utils.autoai.enums import TShirtSize, RunStateTypes, Metrics
from ibm_watson_machine_learning.utils.autoai.utils import chose_model_output

from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, is_cp4d, bucket_exists,\
    bucket_name_gen, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import delete_model_deployment, bucket_cleanup, space_cleanup



class TestAutoAIRemote(unittest.TestCase):
    """
    The test covers:
    - COS set-up (if run on Cloud): checking if bucket exists for the cos instance, if not new bucket is create
    - Saving data to COS
    - regression experiment for xlsx dataset and default (first) sheet
    - downloading all generated pipelines to lale pipeline
    - deployment with lale pipeline (as variable)
    - deployment deletion
    """
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    service_batch: 'Batch' = None
    wml_credentials = None
    cos_credentials = None

    service: 'WebService' = None

    train_data = None

    pipeline_opt: 'RemoteAutoPipelines' = None
    historical_opt: 'RemoteAutoPipelines' = None

    trained_pipeline_details = None
    run_id = None
    random_run_id = None

    best_pipeline = None
    lale_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None

    pod_version = environ.get('KB_VERSION', None)

    data_connection = None
    results_connection = None


    data_location = './autoai/data/xlsx/CarPrice_Assignment.xlsx'

    prediction_column = 'price'
    scoring_method = Metrics.ROOT_MEAN_SQUARED_ERROR

    batch_payload_location = './autoai/data/batch_payload/car_price_payload.csv'

    # CLOUD CONNECTION DETAILS:
    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa")

    cos_endpoint = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
    data_cos_path = 'data/CarPrice_Assignment.xlsx'
    cos_resource_instance_id = None

    results_cos_path = 'results_wml_autoai'

    space_name = 'tests_sdk_space'

    OPTIMIZER_NAME = 'CarPrice wml_autoai regression test'
    DEPLOYMENT_NAME = "CarPrice AutoAI Deployment tests"

    # CP4D CONNECTION DETAILS:
    project_id = None
    space_id = None
    asset_id = None

    

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
        else:
            # note: xlsx is not supported on WMLS
            cls.data_location = './autoai/data/CarPrice_Assignment.csv'
            # --- end note

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
        if self.wml_client.ICP or self.wml_client.WSD:
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

    #################################
    #       REMOTE OPTIMIZER        #
    #################################

    def test_03_initialize_optimizer(self):
        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            desc='test description',
            prediction_type=self.experiment.PredictionType.REGRESSION,
            prediction_column=self.prediction_column,
            scoring=self.scoring_method,
            t_shirt_size=TShirtSize.S if self.wml_client.ICP else TShirtSize.L,
            autoai_pod_version=self.pod_version
        )

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="AutoPipelines did not return RemoteAutoPipelines object")

    def test_04_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    # def test_05_fit_run_training_of_auto_ai_in_wml(self):
    #     failure_count = 0
    #     while failure_count < 4:
    #         try:
    #             TestAutoAIRemote.trained_pipeline_details = self.remote_auto_pipelines.fit(
    #                 training_data_reference=[self.data_connection],
    #                 training_results_reference=self.results_connection,
    #                 background_mode=False)
    #             break
    #         except AutoAIComputeError as e:
    #             import time
    #             failure_count += 1
    #             print(f"Failure {failure_count}\n error:{e}\n\n")
    #             time.sleep(10)
    #
    #     TestAutoAIRemote.run_id = self.trained_pipeline_details['metadata']['id']
    #     self.assertIsInstance(self.trained_pipeline_details, dict,
    #                           msg='Trained pipeline details are not a dictionary instance.')
    #     self.assertTrue(bool(self.trained_pipeline_details))  # Checking if details are not empty.
    #
    #     self.assertIsNotNone(self.data_connection.auto_pipeline_params,
    #                          msg='DataConnection auto_pipeline_params was not updated.')
    #     TestAutoAIRemote.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()
    #     print("train data sample:")
    #     print(self.train_data.head())
    #     self.assertGreater(len(self.train_data), 0)

    def test_05_fit_run_training_of_auto_ai_in_wml_asynch(self):
        TestAutoAIRemote.trained_pipeline_details = self.remote_auto_pipelines.fit(
            training_data_reference=[self.data_connection],
            training_results_reference=self.results_connection,
            background_mode=True)

        TestAutoAIRemote.run_id = self.trained_pipeline_details['metadata']['id']
        self.assertIsNotNone(self.run_id)

    def test_06_get_run_status(self):
        status = self.remote_auto_pipelines.get_run_status()
        self.assertNotEqual(status, RunStateTypes.COMPLETED)

    def test_07_get_run_details(self):
        parameters = self.remote_auto_pipelines.get_run_details()
        # print(parameters)
        self.assertIsNotNone(parameters)

    def test_08_get_summary(self):

        print(f"Run status = {self.remote_auto_pipelines.get_run_status()}")
        # note: check if first pipeline was generated

        metrics = self.wml_client.training.get_details(self.run_id)['entity']['status'].get('metrics', [])
        while chose_model_output("1") not in str(metrics) and self.remote_auto_pipelines.get_run_status() not in ['failed', 'canceled']:
            time.sleep(5)
            print(".", end=" ")
            training_details = self.wml_client.training.get_details(self.run_id)
            metrics = training_details['entity']['status'].get('metrics', [])
        # end note

        print("\n 1st pipeline completed")
        summary_df = self.remote_auto_pipelines.summary()
        print(summary_df)

        self.assertGreater(len(summary_df), 0,
                           msg=f"Summary DataFrame is empty. While {len(metrics)} pipelines are in training_details['entity']['status']['metrics']")

        # check if pipelines are not duplicated
        self.assertEqual(len(summary_df.index.unique()), len(summary_df), msg="Some pipeline names are duplicated in the summary")

    def test_09_get_train_data(self):
        TestAutoAIRemote.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()

        print("train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)

        TestAutoAIRemote.X_values = self.train_data.drop([self.prediction_column], axis=1)[:10].values
        TestAutoAIRemote.y_values = self.train_data[self.prediction_column][:10]

    def test_10_get_best_pipeline_so_far(self):
        best_pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        print(best_pipeline_params)
        print(best_pipeline_params)
        self.assertIn(f"training_{self.scoring_method}", str(best_pipeline_params))

        summary_df = self.remote_auto_pipelines.summary()
        print(summary_df)

        pipeline = self.remote_auto_pipelines.get_pipeline()
        print(f"Fetched pipeline type: {type(pipeline)}")

    def test_11_waiting_for_fitted_completed(self):
        while self.remote_auto_pipelines.get_run_status() == 'running':
            time.sleep(10)

        status = self.remote_auto_pipelines.get_run_status()

        self.assertEqual(status, RunStateTypes.COMPLETED, msg="AutoAI run didn't finished successfully. Status: {}".format(status))

    def test_12_predict_using_fitted_pipeline(self):
        predictions = self.remote_auto_pipelines.predict(X=self.train_data.drop([self.prediction_column], axis=1).values[:5])
        print(predictions)
        self.assertGreater(len(predictions), 0)

    def test_13_summary_listing_all_pipelines_from_wml(self):
        pipelines_details = self.remote_auto_pipelines.summary()
        print(pipelines_details)
        self.assertGreater(len(pipelines_details), 0)

    def test_14_get_pipeline_params_specific_pipeline_parameters(self):
        pipeline_params = self.remote_auto_pipelines.get_pipeline_details(pipeline_name='Pipeline_1')
        print(pipeline_params)
        self.assertIn(f"holdout_{self.scoring_method}", str(pipeline_params))

    def test_15_get_pipeline_params_best_pipeline_parameters(self):
        best_pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        self.assertIn(f"holdout_{self.scoring_method}", str(best_pipeline_params))

    def test_16_get_pipeline_load_specified_pipeline(self):
        TestAutoAIRemote.lale_pipeline = self.remote_auto_pipelines.get_pipeline(
            pipeline_name='Pipeline_1')
        print(f"Fetched pipeline type: {type(self.lale_pipeline)}")
        self.assertIsInstance(self.lale_pipeline, TrainablePipeline)

    ########
    # LALE #
    ########

    def test_17_get_all_pipelines_as_lale(self):
        summary = self.remote_auto_pipelines.summary()
        print(summary)
        failed_pipelines = []
        for pipeline_name in summary.reset_index()['Pipeline Name']:
            print(f"Getting pipeline: {pipeline_name}")
            lale_pipeline = None
            try:
                lale_pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name=pipeline_name)
                self.assertIsInstance(lale_pipeline, TrainablePipeline)
                predictions = lale_pipeline.predict(
                    X=self.train_data.values[:1])
                print(predictions)
                self.assertGreater(len(predictions), 0, msg=f"Returned prediction for {pipeline_name} are empty")
            except:
                print(f"Failure: {pipeline_name}")
                failed_pipelines.append(pipeline_name)
                traceback.print_exc()

            if not TestAutoAIRemote.lale_pipeline:
                TestAutoAIRemote.lale_pipeline = lale_pipeline
                print(f"{pipeline_name} loaded for next test cases")

        self.assertEqual(len(failed_pipelines), 0, msg=f"Some pipelines failed. Full list: {failed_pipelines}")

        #################################
        #      DEPLOYMENT SECTION       #
        #################################

    def test_20_deployment_setup_and_preparation(self):
        TestAutoAIRemote.service = WebService(source_wml_credentials=self.wml_credentials.copy(),
                                              source_project_id=self.project_id,
                                              source_space_id=self.space_id)
        self.wml_client.set.default_space(self.space_id)
        delete_model_deployment(self.wml_client, deployment_name=self.DEPLOYMENT_NAME)
        self.wml_client.set.default_project(self.project_id) if self.project_id else None

    def test_21_deploy_pipepline_by_name(self):
        TestAutoAIRemote.service.create(
            model="Pipeline_4",
            deployment_name=self.DEPLOYMENT_NAME,
            experiment_run_id=self.run_id
        )
        self.assertIsNotNone(self.service.id, msg="Online Deployment creation - missing id")
        self.assertIsNotNone(self.service.name, msg="Online Deployment creation - name not set")
        self.assertIsNotNone(self.service.scoring_url,
                             msg="Online Deployment creation - mscoring url  missing")

    def test_22_score_deployed_model(self):
        nb_records = 5
        predictions = self.service.score(payload=self.train_data.drop([self.prediction_column], axis=1)[:nb_records])
        print(predictions)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions['predictions'][0]['values']), nb_records)

    def test_23_list_deployments(self):
        self.service.list()
        params = self.service.get_params()
        # print(params)
        self.assertIsNotNone(params)

    def test_24_delete_deployment(self):
        print("Delete current deployment: {}".format(self.service.deployment_id))
        self.service.delete()
        self.wml_client.set.default_space(self.space_id) if not self.wml_client.default_space_id else None
        self.wml_client.repository.delete(self.service.asset_id)
        self.wml_client.set.default_project(self.project_id) if is_cp4d() else None

    #########################
    #  Batch deployment
    #########################

    def test_30_batch_deployment_setup_and_preparation(self):
        if self.wml_client.ICP:
            TestAutoAIRemote.service_batch = Batch(source_wml_credentials=self.wml_credentials.copy(),
                                                   source_project_id=self.project_id,
                                                   target_wml_credentials=self.wml_credentials,
                                                   target_space_id=self.space_id)
        else:
            TestAutoAIRemote.service_batch = Batch(self.wml_credentials,
                                                   source_space_id=self.space_id)

        self.assertIsInstance(self.service_batch, Batch, msg="Deployment is not of Batch type.")
        self.assertIsInstance(self.service_batch._source_workspace, WorkSpace, msg="Workspace set incorrectly.")
        self.assertEqual(self.service_batch.id, None, msg="Deployment ID initialized incorrectly")
        self.assertEqual(self.service_batch.name, None, msg="Deployment name initialized incorrectly")

    def test_32__deploy__batch_deploy_pipeline_from_autoai_on_wml(self):
        self.service_batch.create(
            experiment_run_id=self.run_id,
            model="Pipeline_1",
            deployment_name=self.DEPLOYMENT_NAME+' BATCH')

        self.assertIsNotNone(self.service_batch.id, msg="Batch Deployment creation - missing id")
        self.assertIsNotNone(self.service_batch.name, msg="Batch Deployment creation - name not set")
        self.assertIsNotNone(self.service_batch.asset_id,
                          msg="Batch Deployment creation - model (asset) id missing, incorrect model storing")

    def test_33a_run_job_batch_deployed_model_with_data_connection_data_asset(self):
        self.wml_client.set.default_space(self.space_id) if self.wml_client.ICP else None

        asset_details = self.wml_client.data_assets.create(
            name=self.batch_payload_location.split('/')[-1],
            file_path=self.batch_payload_location)
        asset_id = asset_details['metadata']['guid']
        asset_href = self.wml_client.data_assets.get_href(asset_details)

        payload_reference = DataConnection(location=AssetLocation(asset_id=asset_id))

        results_reference = DataConnection(
            location=DeploymentOutputAssetLocation(name="batch_output_car-price.csv"))

        scoring_params = self.service_batch.run_job(
            payload=[payload_reference],
            output_data_reference=results_reference,
            background_mode=False)

        self.wml_client.set.default_project(self.project_id) if self.wml_client.ICP else None

        print(scoring_params)
        self.assertIsNotNone(scoring_params)
        self.wml_client.data_assets.list()

    def test_33b_run_job_batch_deployed_model_with_data_connection_s3(self):
        if self.wml_client.ICP or self.wml_client.WSD:
            self.skipTest("Batch Deployment with s3 data connection is available only for Cloud")
        else:
            payload_reference = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                        secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.batch_payload_location.split('/')[-1])
            )
            results_reference = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                        secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path='batch_output_car-price.csv')
            )
            payload_reference.write(data=self.batch_payload_location, remote_name=self.batch_payload_location.split('/')[-1])

        scoring_params = self.service_batch.run_job(
            payload=[payload_reference],
            output_data_reference=results_reference,
            background_mode=False)
        print(scoring_params)
        self.assertIsNotNone(scoring_params)
        self.wml_client.data_assets.list()

    def test_34_list_batch_deployments(self):
        deployments = self.service_batch.list()
        print(deployments)
        params = self.service_batch.get_params()
        print(params)
        self.assertIsNotNone(params)

    def test_35_delete_deployment_batch(self):
        print("Delete current deployment: {}".format(self.service_batch.deployment_id))
        self.service_batch.delete()
        self.wml_client.set.default_space(self.space_id) if not self.wml_client.default_space_id else None
        self.wml_client.repository.delete(self.service_batch.asset_id)
        self.wml_client.set.default_project(self.project_id) if is_cp4d() else None
        self.assertEqual(self.service_batch.id, None, msg="Deployment ID deleted incorrectly")
        self.assertEqual(self.service_batch.name, None, msg="Deployment name deleted incorrectly")
        self.assertEqual(self.service_batch.scoring_url, None,
                         msg="Deployment scoring_url deleted incorrectly")

if __name__ == '__main__':
    unittest.main()
