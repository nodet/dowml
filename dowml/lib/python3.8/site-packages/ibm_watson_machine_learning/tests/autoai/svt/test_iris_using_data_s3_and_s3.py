import unittest

import ibm_boto3

from ibm_watson_machine_learning.helpers.connections import DataConnection, S3Location, S3Connection
from ibm_watson_machine_learning.tests.utils import bucket_exists, is_cp4d, create_bucket
from ibm_watson_machine_learning.tests.autoai.svt.abstract_test_iris_wml_autoai_multiclass_connections import (
    AbstractTestAutoAIRemote)


@unittest.skipIf(is_cp4d(), "Supported only on CLOUD")
class TestAutoAIRemote(AbstractTestAutoAIRemote, unittest.TestCase):
    """
    The test can be run on CLOUD
    The test covers:
    - COS connection set-up
    - Saving data `iris.csv` to s3 connection
    - downloading training data from connection
    - downloading all generated pipelines to lale pipeline
    - deployment with lale pipeline
    - deployment deletion
    Connection used in test:
     - input: S3 connection pointing to COS.
     - output: S3 connection pointing to COS.
    """

    def test_00b_prepare_COS_instance(self):
        cos_resource = ibm_boto3.resource(
            service_name="s3",
            endpoint_url=self.cos_endpoint,
            aws_access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
            aws_secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
        )
        # Prepare bucket
        if not bucket_exists(cos_resource, self.bucket_name):
            TestAutoAIRemote.bucket_name = create_bucket(cos_resource, self.bucket_name)

        self.assertIsNotNone(TestAutoAIRemote.bucket_name)
        self.assertTrue(bucket_exists(cos_resource, TestAutoAIRemote.bucket_name))

        print(f"Using COS bucket: {TestAutoAIRemote.bucket_name}")

    def test_02_DataConnection_setup(self):
        TestAutoAIRemote.data_connection = DataConnection(
            connection=S3Connection(
                endpoint_url=self.cos_endpoint,
                access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
            ),
            location=S3Location(
                bucket=self.bucket_name,
                path=self.data_cos_path
            )
        )
        TestAutoAIRemote.results_connection = DataConnection(
            connection=S3Connection(
                endpoint_url=self.cos_endpoint,
                access_key_id=self.cos_credentials['cos_hmac_keys']['access_key_id'],
                secret_access_key=self.cos_credentials['cos_hmac_keys']['secret_access_key']
            ),
            location=S3Location(
                bucket=self.bucket_name,
                path=self.results_cos_path
            )
        )
        TestAutoAIRemote.data_connection.write(data=self.data_location, remote_name=self.data_cos_path)

        self.assertIsNotNone(obj=TestAutoAIRemote.data_connection)
        self.assertIsNotNone(obj=TestAutoAIRemote.results_connection)

    def test_02a_read_saved_remote_data_before_fit(self):
        TestAutoAIRemote.data = self.data_connection.read(csv_separator=self.custom_separator)
        print("Data sample:")
        print(self.data.head())
        self.assertGreater(len(self.data), 0)


if __name__ == '__main__':
    unittest.main()
