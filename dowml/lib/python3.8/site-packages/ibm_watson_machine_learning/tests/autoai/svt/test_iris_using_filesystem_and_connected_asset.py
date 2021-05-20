import unittest

import ibm_boto3
from ibm_watson_machine_learning.helpers.connections import DataConnection, FSLocation, S3Location
from ibm_watson_machine_learning.utils.autoai.errors import WMLClientError
from ibm_watson_machine_learning.tests.utils import bucket_exists, is_cp4d
from ibm_watson_machine_learning.tests.autoai.svt.abstract_test_iris_wml_autoai_multiclass_connections import (
    AbstractTestAutoAIRemote)


@unittest.skip("Not ready, training does not support connection_asset")
@unittest.skipIf(not (is_cp4d()), "Supported only on CP4D")
class TestAutoAIRemote(AbstractTestAutoAIRemote, unittest.TestCase):
    """
    The test can be run only on CPD
    The test covers:
    - File system connection set-up
    - downloading training data from filesystem
    - downloading all generated pipelines to lale pipeline
    - deployment with lale pipeline
    - deployment deletion
    Connection used in test:
     - input: File system.
     - output: ConnectedDataAsset pointing to COS.
    """

    def test_00b_prepare_connection_to_COS(self):
        cos_resource = ibm_boto3.resource(
            service_name="s3",
            endpoint_url=self.cos_endpoint,
            aws_access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
            aws_secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
        )

        self.assertIsNotNone(self.bucket_name)
        self.assertTrue(bucket_exists(cos_resource, self.bucket_name))

        connection_details = self.wml_client.connections.create({
            'datasource_type': 'bluemixcloudobjectstorage',
            'name': 'Connection to COS for tests',
            'properties': {
                'bucket': self.bucket_name,
                'access_key': self.cos_credentials['cos_hmac_keys']['access_key_id'],
                'secret_key': self.cos_credentials['cos_hmac_keys']['secret_access_key'],
                'iam_url': self.cos_credentials['iam_url'],
                'url': self.cos_endpoint
            }
        })

        TestAutoAIRemote.connection_id = self.wml_client.connections.get_uid(connection_details)
        self.assertIsInstance(self.connection_id, str)

    def test_02_DataConnection_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(
            location=FSLocation(path=self.data_location)
        )
        TestAutoAIRemote.results_connection = DataConnection(
            connection_asset_id=self.connection_id,
            location=S3Location(
                bucket=self.bucket_name,
                path=self.results_cos_path
            )
        )

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)

    def test_02a_read_saved_remote_data_before_fit(self):
        TestAutoAIRemote.data = self.data_connection.read(csv_separator=self.custom_separator)
        print("Data sample:")
        print(self.data.head())
        self.assertGreater(len(self.data), 0)

    def test_29_delete_connection_and_connected_data_asset(self):
        self.wml_client.connections.delete(self.connection_id)

        with self.assertRaises(WMLClientError):
            self.wml_client.connections.get_details(self.connection_id)


if __name__ == '__main__':
    unittest.main()
