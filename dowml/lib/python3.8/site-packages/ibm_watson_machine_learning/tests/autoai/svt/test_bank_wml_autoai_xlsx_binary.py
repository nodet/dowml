import unittest
from sklearn.pipeline import Pipeline
from pprint import pprint
import pandas as pd
import traceback
from os import environ
import time

from ibm_watson_machine_learning import APIClient

from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import WebService

from ibm_watson_machine_learning.helpers.connections import S3Connection, S3Location, DataConnection, AssetLocation

from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines
from ibm_watson_machine_learning.utils.autoai.errors import CannotReadSavedRemoteDataBeforeFit


from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, is_cp4d, bucket_exists, bucket_name_gen, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import delete_model_deployment, bucket_cleanup, space_cleanup

from ibm_watson_machine_learning.utils.autoai.enums import PositiveLabelClass, RunStateTypes
from ibm_watson_machine_learning.utils.autoai.utils import chose_model_output



from lale import wrap_imported_operators
from lale.operators import TrainablePipeline
from lale.lib.lale import Hyperopt


class TestAutoAIRemote(unittest.TestCase):
    """
    The test covers:
    - COS set-up (if run on Cloud): checking if bucket exists for the cos instance, if not new bucket is create
    - Saving data `xlsx/CarPrice_bank__two_sheets.xlsx` to COS/data assets
    - binary experiment for xlsx file and second sheet described by name: excel_sheet = 'bank' and POSITIVE LABEL
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

    data_location = './autoai/data/xlsx/CarPrice_bank__two_sheets.xlsx'
    sheet_name ='bank'
    sheet_number = 1

    cos_resource_instance_id = None

    trained_pipeline_details = None
    run_id = None

    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-qa")
    pod_version = environ.get('KB_VERSION', None)

    data_connection = None
    results_connection = None

    train_data = None

    cos_endpoint = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
    data_cos_path = 'data/CarPrice_bank__two_sheets.xlsx'

    results_cos_path = 'results_wml_autoai'

    pipeline: 'Pipeline' = None
    lale_pipeline = None
    deployed_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None
    X_values = None
    y_values = None

    OPTIMIZER_NAME = 'Bank xlsx wml_autoai binary test'
    DEPLOYMENT_NAME = "Bank AutoAI Deployment tests"

    scoring_method = PositiveLabelClass.PRECISION_SCORE_MICRO
    max_number_of_estimators = 2
    best_pipeline_name_so_far = None

    project_id = None
    space_id = None

    space_name = 'tests_sdk_space'

    asset_id = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """

        cls.wml_credentials = get_wml_credentials()

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)

        if not cls.wml_client.ICP:
            cls.cos_credentials = get_cos_credentials()
            cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
            cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)
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
            asset_details = self.wml_client.data_assets.create(
                name=self.data_location.split('/')[-1],
                file_path=self.data_location)
            asset_id = asset_details['metadata']['guid']

            TestAutoAIRemote.data_connection = DataConnection(
                location=AssetLocation(asset_id=asset_id))
            TestAutoAIRemote.results_connection = None

        else:  # for cloud and COS
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
        if self.wml_client.ICP:
            with self.assertRaises(CannotReadSavedRemoteDataBeforeFit, msg="Should fail with adequate error"):
                data = self.data_connection.read(excel_sheet=self.sheet_name)

            with self.assertRaises(CannotReadSavedRemoteDataBeforeFit, msg="Should fail with adequate error"):
                data = self.data_connection.read(excel_sheet=self.sheet_number)
        else:
            data = self.data_connection.read(excel_sheet=self.sheet_name)
            print(f"Data sample, excel_sheet ={self.sheet_name}:")
            print(data.head())
            self.assertGreater(len(data), 0)

            data = self.data_connection.read(excel_sheet=self.sheet_number)
            print(f"Data sample, excel_sheet ={self.sheet_number}:")
            print(data.head())
            self.assertGreater(len(data), 0)

    def test_03_initialize_optimizer(self):
        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            desc='test description',
            prediction_type=self.experiment.PredictionType.BINARY,
            prediction_column='y',
            positive_label='yes',
            scoring=self.scoring_method,
            test_size=0.1,
            max_number_of_estimators=self.max_number_of_estimators,
            excel_sheet=self.sheet_name,
            autoai_pod_version=self.pod_version

        )

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    def test_04_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_05_fit_run_training_of_auto_ai_in_wml_asynch(self):
        TestAutoAIRemote.trained_pipeline_details = self.remote_auto_pipelines.fit(
            training_data_reference=[self.data_connection],
            training_results_reference=self.results_connection,
            background_mode=True)

        TestAutoAIRemote.run_id = self.trained_pipeline_details['metadata']['id']
        self.assertIsNotNone(self.run_id)

    def test_06a_get_run_status(self):
        status = self.remote_auto_pipelines.get_run_status()
        self.assertNotEqual(status, RunStateTypes.COMPLETED)

    def test_06b_get_run_details(self):
        parameters = self.remote_auto_pipelines.get_run_details()
        # print(parameters)
        self.assertIsNotNone(parameters)

    def test_07_get_summary(self):

        print(f"Run status = {self.remote_auto_pipelines.get_run_status()}")
        # note: check if first pipeline was generated

        metrics = self.wml_client.training.get_details(self.run_id)['entity']['status'].get('metrics', [])
        while chose_model_output("1") not in str(metrics) and self.remote_auto_pipelines.get_run_status() not in [
            'failed', 'canceled']:
            time.sleep(5)
            print(".", end=" ")
            metrics = self.wml_client.training.get_details(self.run_id)['entity']['status'].get('metrics', [])
        # end note

        print("\n 1st pipeline completed")
        summary_df = self.remote_auto_pipelines.summary()
        print(summary_df)

        self.assertGreater(len(summary_df), 0,
                           msg=f"Summary DataFrame is empty. While {len(metrics)} pipelines are in training_details['entity']['status']['metrics']")

        # check if pipelines are not duplicated
        self.assertEqual(len(summary_df.index.unique()), len(summary_df),
                         msg="Some pipeline names are duplicated in the summary")

    def test_08_get_train_data(self):
        TestAutoAIRemote.train_data = self.remote_auto_pipelines.get_data_connections()[0].read()

        print("train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)

        TestAutoAIRemote.X_values = self.train_data.drop(['y'], axis=1)[:10].values
        TestAutoAIRemote.y_values = self.train_data['y'][:10]

    def test_09_get_best_pipeline_so_far(self):
        best_pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        print(best_pipeline_params)
        self.assertIn(f"holdout_{self.scoring_method}", str(best_pipeline_params))

        summary_df = self.remote_auto_pipelines.summary()
        print(summary_df)

        TestAutoAIRemote.best_pipeline_name_so_far = summary_df.index[0]
        print("\nGetting best calculated pipeline: ", self.best_pipeline_name_so_far)

        pipeline = self.remote_auto_pipelines.get_pipeline()
        print(f"Fetched pipeline type: {type(pipeline)}")

    #################################
    #      DEPLOYMENT SECTION       #
    #################################

    def test_10_deployment_setup_and_preparation(self):
        TestAutoAIRemote.service = WebService(source_wml_credentials=self.wml_credentials.copy(),
                                              source_project_id=self.project_id,
                                              source_space_id=self.space_id)

        self.wml_client.set.default_space(self.space_id)
        delete_model_deployment(self.wml_client, deployment_name=self.DEPLOYMENT_NAME)
        self.wml_client.set.default_project(self.project_id) if self.project_id else None

    def test_11__deploy__online_deploy_pipeline_from_autoai_on_wml(self):
        self.service.create(
            experiment_run_id=self.run_id,
            model=self.best_pipeline_name_so_far,
            deployment_name=self.DEPLOYMENT_NAME + self.best_pipeline_name_so_far)

        self.assertIsNotNone(self.service.id, msg="Online Deployment creation - missing id")
        self.assertIsNotNone(self.service.name, msg="Online Deployment creation - name not set")
        self.assertIsNotNone(self.service.scoring_url,
                             msg="Online Deployment creation - mscoring url  missing")

    def test_12_score_deployed_model(self):
        nb_records = 5
        predictions = self.service.score(payload=self.train_data.drop(['y'], axis=1)[:nb_records])
        print(predictions)
        self.assertIsNotNone(predictions)
        self.assertEqual(len(predictions['predictions'][0]['values']), nb_records)

    def test_13_list_deployments(self):
        self.service.list()
        params = self.service.get_params()
        print(params)
        self.assertIsNotNone(params)

    ###########################################
    #     TRAINING SECTION - FINISH RUN       #
    ###########################################

    def test_15_waiting_for_fitted_completed(self):
        while self.remote_auto_pipelines.get_run_status() == 'running':
            time.sleep(10)

        status = self.remote_auto_pipelines.get_run_status()

        self.assertEqual(status, RunStateTypes.COMPLETED,
                         msg="AutoAI run didn't finished successfully. Status: {}".format(status))

    def test_16_predict_using_fitted_pipeline(self):
        predictions = self.remote_auto_pipelines.predict(X=self.X_values[:5])
        print(predictions)
        self.assertGreater(len(predictions), 0)

    def test_17_summary_listing_all_pipelines_from_wml(self):
        pipelines_details = self.remote_auto_pipelines.summary()
        print(pipelines_details)
        self.assertGreaterEqual(len(pipelines_details), 4*self.max_number_of_estimators)

    def test_18__get_data_connections__return_a_list_with_data_connections_with_optimizer_params(self):
        data_connections = self.remote_auto_pipelines.get_data_connections()
        self.assertIsInstance(data_connections, list, msg="There should be a list container returned")
        self.assertIsInstance(data_connections[0], DataConnection,
                              msg="There should be a DataConnection object returned")

    def test_19_get_pipeline_params_specific_pipeline_parameters(self):
        pipeline_params = self.remote_auto_pipelines.get_pipeline_details(pipeline_name='Pipeline_1')
        print(pipeline_params)

    def test_20__get_pipeline_params__fetch_best_pipeline_parameters__parameters_fetched_as_dict(self):
        best_pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        print(best_pipeline_params)

    ####
    # LALE #
    ########

    def test_21__get_pipeline__load_lale_pipeline__pipeline_loaded(self):
        TestAutoAIRemote.pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name='Pipeline_4')
        print(f"Fetched pipeline type: {type(self.pipeline)}")

        self.assertIsInstance(self.pipeline, TrainablePipeline,
                              msg="Fetched pipeline is not of TrainablePipeline instance.")
        predictions = self.pipeline.predict(
            X=self.X_values[:5])  # works with autoai_libs from branch `development` (date: 9th March 20)
        print(predictions)

    def test_22_get_all_pipelines_as_lale(self):
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
                    X=self.X_values[:1])
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

    ###########################################
    #      DEPLOYMENT SECTION - DELETE        #
    ###########################################
    def test_25_delete_deployment(self):
        print("Delete current deployment: {}".format(self.service.deployment_id))
        self.service.delete()
        self.wml_client.set.default_space(self.space_id) if not self.wml_client.default_space_id else None
        self.wml_client.repository.delete(self.service.asset_id)
        self.wml_client.set.default_project(self.project_id) if is_cp4d() else None


    # @unittest.skip("Skipped lale pretty print")
    # def test_16__pretty_print_lale__checks_if_generated_python_pipeline_code_is_correct(self):
    #     pipeline_code = self.lale_pipeline.pretty_print()
    #     try:
    #         exec(pipeline_code)
    #
    #     except Exception as exception:
    #         self.assertIsNone(exception,
    #                           msg=f"Pretty print from lale pipeline was not successful \n\n Full pipeline code:\n {pipeline_code}")
    #
    # def test_17a_remove_last_freeze_trainable_prefix_returned(self):
    #     from lale.lib.sklearn import KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression
    #     prefix = self.lale_pipeline.remove_last().freeze_trainable()
    #     self.assertIsInstance(prefix, TrainablePipeline,
    #                           msg="Prefix pipeline is not of TrainablePipeline instance.")
    #     TestAutoAIRemote.new_pipeline = prefix >> (KNeighborsClassifier| DecisionTreeClassifier| LogisticRegression)
    #
    # def test_17b_hyperopt_fit_new_pipepiline(self):
    #     hyperopt = Hyperopt(estimator=TestAutoAIRemote.new_pipeline, cv=3, max_evals=2)
    #     try:
    #         TestAutoAIRemote.hyperopt_pipelines = hyperopt.fit(self.X_values, self.y_values)
    #     except ValueError as e:
    #         print(f"ValueError message: {e}")
    #         traceback.print_exc()
    #         hyperopt_results = hyperopt._impl._trials.results
    #         print(hyperopt_results)
    #         self.assertIsNone(e, msg="hyperopt fit was not successful")
    #
    # def test_17c_get_pipeline_from_hyperopt(self):
    #     new_pipeline_model = TestAutoAIRemote.hyperopt_pipelines.get_pipeline()
    #     print(f"Hyperopt_pipeline_model is type: {type(new_pipeline_model)}")
    #     TestAutoAIRemote.new_sklearn_pipeline = new_pipeline_model.export_to_sklearn_pipeline()
    #     self.assertIsInstance(TestAutoAIRemote.new_sklearn_pipeline, Pipeline,
    #                           msg=f"Incorect Sklearn Pipeline type after conversion. Current: {type(TestAutoAIRemote.new_sklearn_pipeline)}")
    #
    # def test_18_predict_refined_pipeline(self):
    #     predictions = TestAutoAIRemote.new_sklearn_pipeline.predict(
    #         X=self.X_values[:1])
    #     print(predictions)
    #     self.assertGreater(len(predictions), 0, msg=f"Returned prediction for refined pipeline are empty")
    #
    #
    # def test_19__get_pipeline__load_specified_pipeline__pipeline_loaded(self):
    #     TestAutoAIRemote.pipeline = self.remote_auto_pipelines.get_pipeline(pipeline_name='Pipeline_4',
    #                                                                         astype=self.experiment.PipelineTypes.SKLEARN)
    #     print(f"Fetched pipeline type: {type(self.pipeline)}")
    #
    #     self.assertIsInstance(self.pipeline, Pipeline,
    #                           msg="Fetched pipeline is not of sklearn.Pipeline instance.")
    #
    # def test_20__predict__do_the_predict_on_sklearn_pipeline__results_computed(self):
    #     predictions = self.pipeline.predict(self.X_values)
    #     print(predictions)


if __name__ == '__main__':
    unittest.main()
