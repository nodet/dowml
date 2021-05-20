import unittest
import logging
import sys
import io
from ibm_watson_machine_learning.helpers.connections import S3Connection, S3Location, DataConnection, AssetLocation, FSLocation
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id, \
    bucket_exists, bucket_name_gen
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup, bucket_cleanup
from ibm_watson_machine_learning import APIClient
from models_preparation import create_tensorflow_model_data


class TestWMLClientWithTensorflow(unittest.TestCase):
    deployment_uid = None
    model_def_uid = None
    scoring_url = None
    cos_resource_instance_id = None
    scoring_data = None
    logger = logging.getLogger(__name__)
    bucket_name = 'tensorflow-training'
    data_cos_path = 'data'
    results_cos_path = 'results'
    data_connections = []
    results_connection = None
    space_name = 'tests_sdk_space'
    data_location = 'svt/datasets/MNIST_DATA'
    model_path = 'svt/artifacts/tf-model-with-metrics_2.1-updated.zip'

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
        TestWMLClientWithTensorflow.space_id = get_space_id(self.wml_client, self.space_name,
                                                 cos_resource_instance_id=self.cos_resource_instance_id)

        # if self.wml_client.ICP:
        #     self.wml_client.set.default_project(self.project_id)
        # else:
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
            TestWMLClientWithTensorflow.bucket_name = bucket_name_gen(prefix=f"{self.bucket_name}-{str(datetime.date.today())}")
            print(f"Creating COS bucket: {TestWMLClientWithTensorflow.bucket_name}")
            cos_resource.Bucket(TestWMLClientWithTensorflow.bucket_name).create()

            self.assertIsNotNone(TestWMLClientWithTensorflow.bucket_name)
            self.assertTrue(bucket_exists(cos_resource, TestWMLClientWithTensorflow.bucket_name))

        print(f"Using COS bucket: {TestWMLClientWithTensorflow.bucket_name}")

    def test_01_save_remote_data_and_DataConnection_setup(self):
        filenames = ['t10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz',
                     'train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz']

        if self.wml_client.ICP:
            for n in filenames:
                asset_details = self.wml_client.data_assets.create(
                    name=n,
                    file_path=self.data_location + '/' + n)
                asset_id = asset_details['metadata']['guid']

                TestWMLClientWithTensorflow.data_connections.append(DataConnection(
                    location=AssetLocation(asset_id=asset_id)))

            TestWMLClientWithTensorflow.results_connection = DataConnection(
                location=FSLocation(path="/spaces/" + str(self.space_id) + "/assets/trainings"))

        else:  # for cloud and COS
            for n in filenames:
                TestWMLClientWithTensorflow.data_connections.append(DataConnection(
                    connection=S3Connection(endpoint_url=self.cos_endpoint,
                                            access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                            secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                    location=S3Location(bucket=self.bucket_name,
                                        path=self.data_cos_path + '/' + n)
                ))

                TestWMLClientWithTensorflow.data_connections[-1].write(data=self.data_location + '/' + n,
                                                                   remote_name=self.data_cos_path + '/' + n)

            TestWMLClientWithTensorflow.results_connection = DataConnection(
                connection=S3Connection(endpoint_url=self.cos_endpoint,
                                        access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                                        secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']),
                location=S3Location(bucket=self.bucket_name,
                                    path=self.results_cos_path)
            )

        self.assertIsNotNone(obj=TestWMLClientWithTensorflow.data_connections)

    def test_02_create_tensorflow_model_definition(self):
        meta_props = {
            self.wml_client.model_definitions.ConfigurationMetaNames.NAME: "TF 2.1 Model Definition NB",
            self.wml_client.model_definitions.ConfigurationMetaNames.DESCRIPTION: "SVT Model Def Tensorflow",
            self.wml_client.model_definitions.ConfigurationMetaNames.VERSION: "1.0",
            self.wml_client.model_definitions.ConfigurationMetaNames.PLATFORM: {"name": "python", "versions": ["3.7"]}
        }

        model_def_details = self.wml_client.model_definitions.store(self.model_path, meta_props)
        TestWMLClientWithTensorflow.model_def_uid = self.wml_client.model_definitions.get_uid(model_def_details)

    def test_03_create_and_run_training(self):
        metadata = {
            self.wml_client.training.ConfigurationMetaNames.NAME: "Tensorflow Training from Notebook",
            self.wml_client.training.ConfigurationMetaNames.DESCRIPTION: "",
            self.wml_client.training.ConfigurationMetaNames.TAGS: [
                {
                    "value": "pyclienttraining",
                    "description": "python client training"
                }
            ],
            self.wml_client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [x._to_dict() for x in TestWMLClientWithTensorflow.data_connections],
            self.wml_client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: TestWMLClientWithTensorflow.results_connection._to_dict(),
            self.wml_client.training.ConfigurationMetaNames.MODEL_DEFINITION: {
                "id": self.model_def_uid,
                "command": "python convolutional_network.py --trainImagesFile train-images-idx3-ubyte.gz --trainLabelsFile train-labels-idx1-ubyte.gz --testImagesFile t10k-images-idx3-ubyte.gz --testLabelsFile t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000",
                "software_spec": {
                    "name": "tensorflow_2.1-py3.7"
                },
                "hardware_spec": {
                    "name": "K80",
                    "num_nodes": 1
                },
                "parameters": {
                    "name": "TF 2.1 Training from Notebook",
                    "description": "TF 2.1 training from Python Client notebook"

                }
            }
        }

        training_details = self.wml_client.training.run(meta_props=metadata, asynchronous=False)
        self.assertEqual(training_details['entity']['status']['state'], 'completed')

    def test_04_delete_model_definition(self):
        TestWMLClientWithTensorflow.logger.info("Delete model definition")
        self.wml_client.repository.delete(TestWMLClientWithTensorflow.model_def_uid)


if __name__ == '__main__':
    unittest.main()
