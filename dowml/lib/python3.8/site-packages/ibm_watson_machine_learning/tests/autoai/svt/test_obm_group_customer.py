import unittest
from sklearn.pipeline import Pipeline
from pprint import pprint
import pandas as pd
import traceback
from os import environ

from ibm_watson_machine_learning import APIClient

from ibm_watson_machine_learning.experiment import AutoAI
from ibm_watson_machine_learning.deployment import Batch
from ibm_watson_machine_learning.preprocessing import DataJoinGraph
from ibm_watson_machine_learning.preprocessing.data_join_pipeline import OBMPipelineGraphBuilder

from ibm_watson_machine_learning.helpers.connections import S3Connection, S3Location, DataConnection, FSLocation

from ibm_watson_machine_learning.experiment.autoai.optimizers import RemoteAutoPipelines

from ibm_watson_machine_learning.tests.utils import bucket_exists, \
    bucket_name_gen, get_space_id

from ibm_watson_machine_learning.tests.utils.cleanup import bucket_cleanup, space_cleanup

from ibm_watson_machine_learning.tests.utils import (
    get_wml_credentials, get_cos_credentials, is_cp4d, print_test_separators)
from ibm_watson_machine_learning.utils.autoai.errors import WrongDataJoinGraphNodeName

get_step_details = OBMPipelineGraphBuilder.get_step_details
get_join_extractors = OBMPipelineGraphBuilder.get_join_extractors
get_extractor_columns = OBMPipelineGraphBuilder.get_extractor_columns
get_extractor_transformations = OBMPipelineGraphBuilder.get_extractor_transformations


class TestOBMOnly(unittest.TestCase):
    data_join_graph: 'DataJoinGraph' = None
    wml_client: 'APIClient' = None
    experiment: 'AutoAI' = None
    remote_auto_pipelines: 'RemoteAutoPipelines' = None
    wml_credentials = None
    cos_credentials = None
    pipeline_opt: 'RemoteAutoPipelines' = None
    historical_opt: 'RemoteAutoPipelines' = None
    service: 'Batch' = None
    data_join_pipeline: 'DataJoinPipeline' = None

    data_location = './autoai/data/'

    trained_pipeline_details = None
    run_id = None

    data_connections = []
    main_conn = None
    customers_conn = None
    transactions_conn = None
    purchases_conn = None
    products_conn = None
    results_connection = None

    train_data = None
    holdout_data = None
    bucket_name = environ.get('BUCKET_NAME', "wml-autoaitests-obm-qa")

    cos_endpoint =  "https://s3.us-south.cloud-object-storage.appdomain.cloud"
    cos_resource_instance_id = None
    data_cos_path = 'data/'

    results_cos_path = 'results_wml_autoai'

    pipeline: 'Pipeline' = None
    lale_pipeline = None
    deployed_pipeline = None
    hyperopt_pipelines = None
    new_pipeline = None
    new_sklearn_pipeline = None

    OPTIMIZER_NAME = 'OBM only test'
    DEPLOYMENT_NAME = "OBM only deployment test"

    space_name = 'tests_sdk_space'

    job_id = None

    # # CP4D CONNECTION DETAILS:
    # if 'qa' in get_env().lower():
    #     space_id = 'aa81ee79-795a-4a23-8265-e87966dfe437'
    #     project_id = '3d177338-3f8b-444d-b385-9ff929fd124f'
    # else:
    #     project_id = '051e12c2-a65d-49d7-8ec3-04e3c36a60eb'
    #     space_id = None

    project_id = None
    space_id = None
    asset_id = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """
        cls.wml_credentials = get_wml_credentials()
        if not is_cp4d():
            cls.cos_credentials = get_cos_credentials()
            if 'endpoint_url' in cls.cos_credentials:
                cls.cos_endpoint = cls.cos_credentials['endpoint_url']
            cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials.copy())
        cls.project_id = cls.wml_credentials.get('project_id')

    @print_test_separators
    def test_00a_space_cleanup(self):
        space_cleanup(self.wml_client,
                      get_space_id(self.wml_client, self.space_name,
                                   cos_resource_instance_id=self.cos_resource_instance_id),
                      days_old=7)
        TestOBMOnly.space_id = get_space_id(self.wml_client, self.space_name,
                                     cos_resource_instance_id=self.cos_resource_instance_id)

        if self.wml_client.ICP:
            self.wml_client.set.default_project(self.project_id)
        else:
            self.wml_client.set.default_space(self.space_id)

    @print_test_separators
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
            TestOBMOnly.bucket_name = bucket_name_gen(prefix=f"{self.bucket_name}-{str(datetime.date.today())}")
            print(f"Creating COS bucket: {TestOBMOnly.bucket_name}")
            cos_resource.Bucket(TestOBMOnly.bucket_name).create()

            self.assertIsNotNone(TestOBMOnly.bucket_name)
            self.assertTrue(bucket_exists(cos_resource, TestOBMOnly.bucket_name))

        print(f"Using COS bucket: {TestOBMOnly.bucket_name}")

    @print_test_separators
    def test_01_create_multiple_data_connections__connections_created(self):
        print("Creating multiple data connections...")
        if self.wml_client.ICP:
            TestOBMOnly.main_conn = DataConnection(
                data_join_node_name="main",
                location=FSLocation(path=self.data_location + 'group_customer_main.csv'))

            TestOBMOnly.customers_conn = DataConnection(
                data_join_node_name="customers",
                location=FSLocation(path=self.data_location + 'group_customer_customers.csv'))

            TestOBMOnly.transactions_conn = DataConnection(
                data_join_node_name="transactions",
                location=FSLocation(path=self.data_location + 'group_customer_transactions.csv'))

            TestOBMOnly.purchases_conn = DataConnection(
                data_join_node_name="purchases",
                location=FSLocation(path=self.data_location + 'group_customer_purchase.csv'))

            TestOBMOnly.products_conn = DataConnection(
                data_join_node_name="products",
                location=FSLocation(path=self.data_location + 'group_customer_products.csv'))

            TestOBMOnly.results_connection = None
        else:
            TestOBMOnly.main_conn = DataConnection(
                data_join_node_name="main",
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'group_customer_main.csv'))

            TestOBMOnly.customers_conn = DataConnection(
                data_join_node_name="customers",
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'group_customer_customers.csv'))

            TestOBMOnly.transactions_conn = DataConnection(
                data_join_node_name="transactions",
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'group_customer_transactions.csv'))

            TestOBMOnly.purchases_conn = DataConnection(
                data_join_node_name="purchases",
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'group_customer_purchase.csv'))

            TestOBMOnly.products_conn = DataConnection(
                data_join_node_name="products",
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'group_customer_products.csv'))

            TestOBMOnly.results_connection = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                        secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.results_cos_path)
            )

        TestOBMOnly.data_connections = [self.main_conn,
                                        self.customers_conn,
                                        self.transactions_conn,
                                        self.purchases_conn,
                                        self.products_conn]

    @print_test_separators
    def test_02_create_data_join_graph__graph_created(self):
        print("Defining DataJoinGraph...")
        data_join_graph = DataJoinGraph()
        data_join_graph.node(name="main")
        data_join_graph.node(name="customers")
        data_join_graph.node(name="transactions")
        data_join_graph.node(name="purchases")
        data_join_graph.node(name="products")
        data_join_graph.edge(from_node="main", to_node="customers",
                             from_column=["group_customer_id"], to_column=["group_customer_id"])
        data_join_graph.edge(from_node="main", to_node="transactions",
                             from_column=["transaction_id"], to_column=["transaction_id"])
        data_join_graph.edge(from_node="main", to_node="purchases",
                             from_column=["group_id"], to_column=["group_id"])
        data_join_graph.edge(from_node="transactions", to_node="products",
                             from_column=["product_id"], to_column=["product_id"])

        TestOBMOnly.data_join_graph = data_join_graph

        print(f"data_join_graph: {data_join_graph}")

    # @print_test_separators
    # def test_03_data_join_graph_visualization(self):
    #     print("Visualizing data_join_graph...")
    #     self.data_join_graph.visualize()

    @print_test_separators
    def test_04_initialize_AutoAI_experiment__pass_credentials__object_initialized(self):
        print("Initializing AutoAI experiment...")
        TestOBMOnly.experiment = AutoAI(wml_credentials=self.wml_credentials.copy(),
                                        project_id=self.project_id,
                                        space_id=self.space_id)

        self.assertIsInstance(self.experiment, AutoAI, msg="Experiment is not of type AutoAI.")

    @print_test_separators
    def test_05_save_remote_data_and_DataConnection_setup(self):
        print("Writing multiple data files to COS...")
        if is_cp4d():
            pass

        else:  # for cloud and COS
            self.main_conn.write(data=self.data_location + 'group_customer_main.csv',
                                 remote_name=self.data_cos_path + 'group_customer_main.csv')
            self.customers_conn.write(data=self.data_location + 'group_customer_customers.csv',
                                      remote_name=self.data_cos_path + 'group_customer_customers.csv')
            self.transactions_conn.write(data=self.data_location + 'group_customer_transactions.csv',
                                         remote_name=self.data_cos_path + 'group_customer_transactions.csv')
            self.purchases_conn.write(data=self.data_location + 'group_customer_purchase.csv',
                                      remote_name=self.data_cos_path + 'group_customer_purchase.csv')
            self.products_conn.write(data=self.data_location + 'group_customer_products.csv',
                                     remote_name=self.data_cos_path + 'group_customer_products.csv')

    @print_test_separators
    def test_06_initialize_optimizer(self):
        print("Initializing optimizer with data_join_graph...")
        TestOBMOnly.remote_auto_pipelines = self.experiment.optimizer(
            name=self.OPTIMIZER_NAME,
            prediction_type=AutoAI.PredictionType.REGRESSION,
            prediction_column='next_purchase',
            scoring=AutoAI.Metrics.LOG_LOSS,
            t_shirt_size=self.experiment.TShirtSize.L if not is_cp4d() else self.experiment.TShirtSize.S,
            data_join_graph=self.data_join_graph,
            data_join_only=True,
        )

        self.assertIsInstance(self.remote_auto_pipelines, RemoteAutoPipelines,
                              msg="experiment.optimizer did not return RemoteAutoPipelines object")

    @print_test_separators
    def test_07_get_configuration_parameters_of_remote_auto_pipeline(self):
        print("Getting experiment configuration parameters...")
        parameters = self.remote_auto_pipelines.get_params()
        print(parameters)
        self.assertIsInstance(parameters, dict, msg='Config parameters are not a dictionary instance.')

    def test_08a_check_data_join_node_names(self):
        if self.wml_client.ICP:
            main_conn = DataConnection(
                location=FSLocation(path=self.data_location + 'group_customer_main.csv'))

            customers_conn = DataConnection(
                location=FSLocation(path=self.data_location + 'group_customer_customers.csv'))

            transactions_conn = DataConnection(
                location=FSLocation(path=self.data_location + 'group_customer_transactions.csv'))

            purchases_conn = DataConnection(
                location=FSLocation(path=self.data_location + 'group_customer_purchase.csv'))

            products_conn = DataConnection(
                location=FSLocation(path=self.data_location + 'group_customer_products.csv'))
        else:
            main_conn = DataConnection(
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'group_customer_main.csv'))

            customers_conn = DataConnection(
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'group_customer_customers.csv'))

            transactions_conn = DataConnection(
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'group_customer_transactions.csv'))

            purchases_conn = DataConnection(
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'group_customer_purchase.csv'))

            products_conn = DataConnection(
                connection=S3Connection(
                    endpoint_url=self.cos_endpoint,
                    access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                    secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.data_cos_path + 'group_customer_products.csv'))

        data_connections = [main_conn,
                            customers_conn,
                            transactions_conn,
                            purchases_conn,
                            products_conn]

        with self.assertRaises(WrongDataJoinGraphNodeName, msg="Training started with wrong data join node names"):
            self.remote_auto_pipelines.fit(
                training_data_reference=data_connections,
                training_results_reference=self.results_connection,
                background_mode=False)

    @print_test_separators
    def test_08b_fit_run_training_of_auto_ai_in_wml(self):
        print("Scheduling OBM only training...")
        TestOBMOnly.trained_pipeline_details = self.remote_auto_pipelines.fit(
            training_data_reference=self.data_connections,
            training_results_reference=self.results_connection,
            background_mode=False)

        TestOBMOnly.run_id = self.trained_pipeline_details['metadata']['id']

        for connection in self.data_connections:
            self.assertIsNotNone(
                connection.auto_pipeline_params,
                msg=f'DataConnection auto_pipeline_params was not updated for connection: {connection.id}')

    @print_test_separators
    def test_09_download_original_training_data(self):
        print("Downloading each training file...")
        for connection in self.remote_auto_pipelines.get_data_connections():
            train_data = connection.read()

            print(f"Connection: {connection.id} - train data sample:")
            print(train_data.head())
            self.assertGreater(len(train_data), 0)

    @print_test_separators
    def test_10_download_preprocessed_obm_training_data(self):
        print("Downloading OBM preprocessed training data with holdout split...")
        TestOBMOnly.train_data, TestOBMOnly.holdout_data = (
            self.remote_auto_pipelines.get_preprocessed_data_connection().read(with_holdout_split=True))

        print("OBM train data sample:")
        print(self.train_data.head())
        self.assertGreater(len(self.train_data), 0)
        print("OBM holdout data sample:")
        print(self.holdout_data.head())
        self.assertGreater(len(self.holdout_data), 0)

    @print_test_separators
    def test_11_visualize_obm_pipeline(self):
        if self.wml_client.ICP or self.wml_client.WSD:
            self.skipTest("Visualizing pipeline not supported on CPD")

        print("Visualizing OBM model ...")
        TestOBMOnly.data_join_pipeline = self.remote_auto_pipelines.get_preprocessing_pipeline()
        assert isinstance(self.data_join_pipeline._pipeline_json, list)
        assert '40 [label=NumberSet]' in self.data_join_pipeline._graph.__str__()
    #     self.data_join_pipeline.visualize()

    @print_test_separators
    def test_12_check_if_data_join_pipeline_graph_correct(self):
        if self.wml_client.ICP or self.wml_client.WSD:
            self.skipTest("Visualizing pipeline not supported on CPD")
        pipeline_json = self.data_join_pipeline._pipeline_json
        graph = self.data_join_pipeline._graph_json

        step_types = [message['feature_engineering_components']['obm'][0]['step_type'] for message in pipeline_json]
        last_non_join_iteration = step_types.index('join')
        selection_iteration = step_types.index('feature selection') + 1
        join_iterations = [i + 1 for i, x in enumerate(step_types) if x == "join"]

        for message in pipeline_json:
            name, iteration, _ = get_step_details(message)
            self.assertTrue(str(iteration) in graph['nodes'])

            if 1 < iteration <= 2:
                self.assertTrue(str(iteration) in graph['edges'][str(iteration - 1)])
            elif iteration in join_iterations:
                self.assertTrue(str(iteration) in graph['edges'][str(last_non_join_iteration)])
                extractors = get_join_extractors(message)

                if extractors is None:
                    continue
                for ext, i in zip(extractors, range(len(extractors))):
                    ext_index = str(iteration) + str(i)
                    self.assertTrue(ext_index in graph['nodes'] and
                                    ext_index in ext_index in graph['edges'][str(iteration)])

                    columns = get_extractor_columns(extractors[ext])
                    transformations = get_extractor_transformations(extractors[ext])
                    for j, column in enumerate(columns):
                        col_index = str(iteration) + str(i) + str(j)
                        self.assertTrue(col_index in graph['nodes'] and col_index in graph['edges'][str(ext_index)])

                        for transformation in transformations:
                            self.assertTrue(transformation in graph['edges'][str(col_index)])
                            self.assertTrue(str(selection_iteration) in graph['edges'][str(transformation)])

            elif iteration > selection_iteration:
                self.assertTrue(str(iteration) in graph['edges'][str(iteration - 1)])

    @print_test_separators
    def test_13_get_run_status(self):
        print("Getting training status...")
        status = self.remote_auto_pipelines.get_run_status()
        print(status)
        self.assertEqual("completed", status, msg="AutoAI run didn't finished successfully. Status: {}".format(status))

    @print_test_separators
    def test_14_get_run_details(self):
        print("Getting training details...")
        parameters = self.remote_auto_pipelines.get_run_details()
        print(parameters)
        self.assertIsNotNone(parameters)

    @print_test_separators
    def test_15__get_data_connections__return_a_list_with_data_connections_with_optimizer_params(self):
        print("Getting all data connections...")
        data_connections = self.remote_auto_pipelines.get_data_connections()
        self.assertIsInstance(data_connections, list, msg="There should be a list container returned")
        self.assertIsInstance(data_connections[0], DataConnection,
                              msg="There should be a DataConnection object returned")

    @print_test_separators
    def test_16_get_historical_optimizer_with_data_join_graph(self):
        print("Fetching historical optimizer with OBM only...")
        TestOBMOnly.historical_opt = self.experiment.runs.get_optimizer(run_id=self.run_id)
        self.assertIsInstance(self.historical_opt.get_params()['data_join_graph'], DataJoinGraph,
                              msg="data_join_graph was incorrectly loaded")

    @print_test_separators
    def test_17_get_obm_training_data_from_historical_optimizer(self):
        print("Download historical preprocessed data from OBM stage...")
        train_data = self.historical_opt.get_preprocessed_data_connection().read()

        print("OBM train data sample:")
        print(train_data.head())
        self.assertGreater(len(train_data), 0)


if __name__ == '__main__':
    unittest.main()
