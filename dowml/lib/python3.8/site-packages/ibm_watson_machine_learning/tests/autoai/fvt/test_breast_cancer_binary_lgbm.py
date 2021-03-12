import unittest
from sklearn.pipeline import Pipeline
from pprint import pprint
import pandas as pd

from ibm_watson_machine_learning import APIClient

from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import WebService

from ibm_watson_machine_learning.helpers.connections import S3Connection, S3Location, DataConnection, AssetLocation

from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines

from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, is_cp4d, get_env
from ibm_watson_machine_learning.tests.utils.cleanup import delete_model_deployment

# from tests.utils import get_wml_credentials, get_cos_credentials, is_cp4d, get_env
# from tests.utils.cleanup import delete_model_deployment

from lale.operators import TrainablePipeline

class TestAutoAIRemote(unittest.TestCase):
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None

    service: 'WebService' = None

    train_data = None
    holdout_data = None

    pipeline_opt: 'RemoteAutoPipelines' = None
    historical_opt: 'RemoteAutoPipelines' = None

    trained_pipeline_details = None
    run_id = None
    random_run_id = None

    best_pipeline = None
    lale_pipeline = None

    data_connection = None
    results_connection = None

    data_location = './autoai/data/breast_cancer.csv'

    # CLOUD CONNECTION DETAILS:

    bucket_name = "pycharm"
    data_cos_path = 'data/breast_cancer.csv'

    results_cos_path = 'results_wml_autoai'

    OPTIMIZER_NAME = 'CarPrice wml_autoai regression test'
    DEPLOYMENT_NAME = "CarPrice AutoAI Deployment tests"

    # CP4D CONNECTION DETAILS:

    # svt17
    # space_id = 'd651b5f4-1fa0-40de-9e80-cebd5098b57a'
    # project_id = '49030f0d-84b5-4a1a-a335-9aae03ca6f10'
    # asset_id = None

    # WML77
    project_id = '34005267-5e70-4253-96bb-21e82bcb60a2'
    space_id = '776e5097-41ad-4724-bda8-9b070c9c835f'
    asset_id = None

    @classmethod
    def setUp(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.data = pd.read_csv(cls.data_location)
        cls.X = cls.data.drop(['diagnosis'], axis=1)
        cls.y = cls.data['diagnosis']

        cls.wml_credentials = get_wml_credentials()

        if not is_cp4d():
            cls.cos_credentials = get_cos_credentials()

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)

    def test_01_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        if is_cp4d():
            TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials,
                                                 project_id=self.project_id,
                                                 space_id=self.space_id)
        else:
            TestAutoAIRemote.experiment = AutoAI(wml_credentials=self.wml_credentials)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    def test_02_save_remote_data_and_DataConnection_setup(self):
        if is_cp4d():
            asset_details = self.wml_client.data_assets.create(
                name="breast_cancer.csv",
                file_path=self.data_location)
            asset_id = asset_details['metadata']['guid']

            TestAutoAIRemote.data_connection = DataConnection(
                location=AssetLocation(asset_id=asset_id))
            TestAutoAIRemote.results_connection = None

        else:  # for cloud and COS
            TestAutoAIRemote.data_connection = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_credentials['endpoint_url'],
                                        access_key_id=self.cos_credentials['access_key_id'],
                                        secret_access_key=self.cos_credentials['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path)
            )
            TestAutoAIRemote.results_connection = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_credentials['endpoint_url'],
                                        access_key_id=self.cos_credentials['access_key_id'],
                                        secret_access_key=self.cos_credentials['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.results_cos_path)
            )
            TestAutoAIRemote.data_connection.write(data=self.data, remote_name=self.data_cos_path)

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)

    #################################
    #       REMOTE OPTIMIZER        #
    #################################

    def test_03_initialize_optimizer(self):
        TestAutoAIRemote.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            desc='test description',
            prediction_type=self.experiment.PredictionType.BINARY,
            prediction_column='diagnosis',
            scoring=self.experiment.Metrics.ROC_AUC_SCORE,
            daub_include_only_estimators=[self.experiment.ClassificationAlgorithms.LGBM]
        )
        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="AutoPipelines did not return RemoteAutoPipelines object")

    def test_04_get_configuration_parameters_of_remote_auto_pipeline(self):
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_05_fit_run_training_of_auto_ai_in_wml(self):
        TestAutoAIRemote.trained_pipeline_details = self.remote_auto_pipelines.fit(
            training_data_reference=[self.data_connection],
            training_results_reference=self.results_connection,
            background_mode=False)

        TestAutoAIRemote.run_id = self.trained_pipeline_details['metadata']['id']
        self.assertIsInstance(self.trained_pipeline_details, dict,
                              msg='Trained pipeline details are not a dictionary instance.')
        self.assertTrue(bool(self.trained_pipeline_details))  # Checking if details are not empty.

        self.assertIsNotNone(self.data_connection.auto_pipeline_params,
                             msg='DataConnection auto_pipeline_params was not updated.')
        TestAutoAIRemote.train_data, TestAutoAIRemote.holdout_data = self.remote_auto_pipelines.get_data_connections()[
            0].read(with_holdout_split=True)
        print("train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)
        print("holdout data sample:")
        print(self.holdout_data.head())
        self.assertGreater(len(self.holdout_data), 0)

    def test_06_get_run_status(self):
        status = self.remote_auto_pipelines.get_run_status()
        self.assertEqual(status, "completed", msg="AutoAI run didn't finished successfully. Status: {}".format(status))

    def test_07_get_run_details(self):
        parameters = self.remote_auto_pipelines.get_run_details()
        # print(parameters)

    def test_08_predict_using_fitted_pipeline(self):
        predictions = self.remote_auto_pipelines.predict(X=self.holdout_data.drop(['diagnosis'], axis=1).values[:5])
        print(predictions)
        self.assertGreater(len(predictions), 0)

    def test_09_summary_listing_all_pipelines_from_wml(self):
        pipelines_details = self.remote_auto_pipelines.summary()
        print(pipelines_details)

    def test_10_get_pipeline_params_specific_pipeline_parameters(self):
        pipeline_params = self.remote_auto_pipelines.get_pipeline_details(pipeline_name='Pipeline_1')
        # print(pipeline_params)

    def test_11_get_pipeline_params_best_pipeline_parameters(self):
        best_pipeline_params = self.remote_auto_pipelines.get_pipeline_details()
        # print(best_pipeline_params)

    def test_12_get_pipeline_load_specified_pipeline(self):
        TestAutoAIRemote.lale_pipeline = self.remote_auto_pipelines.get_pipeline(
            pipeline_name='Pipeline_1')
        print(f"Fetched pipeline type: {type(self.lale_pipeline)}")
        self.assertIsInstance(self.lale_pipeline, TrainablePipeline)
        # predictions = self.lale_pipeline.predict(
        #     X=self.holdout_data.drop(['diagnosis'], axis=1).values[:5])  # works with autoai_libs from branch `development` (date: 9th March 20)
        # print(predictions)

        #################################
        #      DEPLOYMENT SECTION       #
        #################################
    # @unittest.skip("Skipping")
    def test_13_pretty_print_lale_checks_if_generated_python_pipeline_code_is_correct(self):
        base_string: str = "\
import pandas as pd\r\n\
from sklearn.model_selection import train_test_split\r\n\
from sklearn.metrics import  accuracy_score\r\n\
data=pd.read_csv('{0}')\r\n\
X = data.drop(['diagnosis'], axis=1).values\r\n\
y = data['diagnosis'].values\r\n\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)".format(
            self.data_location)

        end_string = "\
pipeline.fit(X_train,y_train)\r\n\
print(accuracy_score(y_test,pipeline.predict(X_test)))"

        pipeline_code = TestAutoAIRemote.lale_pipeline.pretty_print(combinators=False)

        exception = None
        try:
            print(base_string + "\r\n" + pipeline_code + "\r\n" + end_string)
            exec(base_string + "\r\n" + pipeline_code + "\r\n" + end_string)
        except Exception as exception:
            self.assertIsNone(exception, msg="Pretty print from lale pipeline was not successful")

if __name__ == '__main__':
    unittest.main()
